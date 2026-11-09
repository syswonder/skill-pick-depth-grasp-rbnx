# Description: 从多机位数据里拟合臂的几何修正（关节零点 + 各关节的小平移）。
#
# 原理：板子在整个采集过程里没动过，所以每个位姿算出来的"板在基座系"的位置本该完全一样。
# 差异从哪来？法兰位姿是拿关节角走模型算的，模型和实物不一致，这块差异就留在板子的散布里。
# 把模型参数当未知量放开，散布能掉多少、参数落在什么量级，就是"实物和标称差多少"的答案。
#
# 数据必须跨机位：单机位下模型参数和相机外参 X 完全退化——改一个连杆长度，法兰整体挪一下，
# X 跟着挪一下就对消了，残差一点不变。这一点已经在真实数据上验过，不是理论担心。
#
# 拟合出来的东西必须过三关才准用（只过第一关等于自欺）：
#   一、散布确实掉下来；
#   二、参数落在物理讲得通的量级里；
#   三、分半验证——用一半位姿拟合，去预测没参与的另一半，也得准。
# 三关里任何一关不过，就不要这份修正：宁可维持现状，也不要一个"拟合得很好但不知道在拟合什么"的模型。
import numpy as np
from scipy.optimize import least_squares

from common import handeye3d
from common.robot import kinematics
from common.robot.kinematics import JOINT_COUNT

# 旋转残差的换算：把弧度折成毫米量级，好让位置和姿态两部分在同一个尺度上被优化。
# 取 1 rad ≈ 0.3 m——一个一弧度的姿态错，离原点 0.3 m 的点就挪 0.3 m，量级相当。
ROTATION_TO_MM = 300.0
MIN_POSES = 12                  # 少于这么多位姿不拟合
MIN_SPREAD_DEG = 25.0           # 关节角跨度小于这个数，模型参数定不住
MIN_IMPROVEMENT = 0.35          # 分半验证里，检验集的散布至少要掉这么多才算数

# 放哪些参数，是按"数据能不能定住它"剪过的，不是随便挑的：
#   第 1 个关节的平移 = 整条臂在基座系里整体挪一下，和"相机外参 X 挪一下"完全等价，被 X 吸收；
#   第 6 个关节的零点与平移 = 整段落在法兰系里的常数，同样被 X 吸收，而且和「夹爪尖」那页
#   标的偏移本来就是同一件事。
# 留着它们不会得到更多信息，只会让拟合在零空间里乱跑——实测跑出过 5.8 米的平移。
# 真正能定住的是手腕的转角和中间几节的位置，这也正是实臂误差的主要来源。
ZERO_JOINTS = (0, 1, 2, 3, 4)
OFFSET_JOINTS = (1, 2, 3, 4)
PARAM_COUNT = len(ZERO_JOINTS) + 3 * len(OFFSET_JOINTS)
# 正则项：每个参数按"到物理上限的比例"罚一点。定得住的参数，这点罚可以忽略；
# 定不住的方向数据本来就没有收益，于是被稳稳钉在零。
REGULARIZATION = 1.0


def _kinpy_chain(content, end_link):
    """解析 URDF 文本成串联链。

    先去掉 XML 声明：kinpy 底下用 lxml，带编码声明的 unicode 字符串它不认
    （同 common/robot/ik.py 里的那段处理）。
    """
    import re

    import kinpy

    return kinpy.build_serial_chain_from_urdf(re.sub(r"<\?xml[^?]*\?>", "", content, count=1), end_link)


class ArmModel:
    """一条臂的运动学模型，能按给定的平移修正重建。

    零点不重建模型：把零点加到关节角上再走正运动学，和"改关节零位"是同一件事，
    而且不用动 URDF。
    """

    def __init__(self, content: str, end_link: str):
        self.content = content
        self.end_link = end_link
        self._nominal = _kinpy_chain(content, end_link)
        # 关节名按链序（从基座数起），平移修正要靠它对应到 URDF 里的哪一个 joint
        self.joint_names = list(self._nominal.get_joint_parameter_names())

    def forward(self, offsets_m=None):
        """给一组关节平移，返回一个 fk(joints_deg) -> 4x4 的可调用对象。"""
        offsets = np.zeros((JOINT_COUNT, 3)) if offsets_m is None else np.asarray(offsets_m, dtype=float)
        if np.allclose(offsets, 0.0):
            chain = self._nominal
        else:
            content = kinematics._insert_offsets(self.content, list(self.joint_names), offsets)
            chain = _kinpy_chain(content, self.end_link)
        return lambda joints_deg: np.asarray(chain.forward_kinematics(
            np.deg2rad(np.asarray(joints_deg, dtype=float))).matrix())


def unpack(params) -> dict:
    """参数向量 → 修正字典；没参与拟合的关节保持零。"""
    params = np.asarray(params, dtype=float)
    correction = kinematics.identity()
    for slot, joint in enumerate(ZERO_JOINTS):
        correction["zeros_deg"][joint] = float(params[slot])
    offsets = params[len(ZERO_JOINTS):].reshape(len(OFFSET_JOINTS), 3)
    for slot, joint in enumerate(OFFSET_JOINTS):
        correction["offsets_m"][joint] = offsets[slot].tolist()
    correction["source"] = "fit"
    return correction


def pack(correction) -> np.ndarray:
    """修正字典 → 参数向量。"""
    zeros = np.asarray(correction.get("zeros_deg", [0.0] * JOINT_COUNT), dtype=float)
    offsets = np.asarray(correction.get("offsets_m", [[0.0] * 3] * JOINT_COUNT), dtype=float)
    return np.r_[zeros[list(ZERO_JOINTS)], offsets[list(OFFSET_JOINTS)].ravel()]


def _regularization(params) -> np.ndarray:
    """每个参数按"到物理上限的比例"罚一点，把零空间钉在零。

    没有这一项时，数据管不着的方向残差恒为零，优化器可以随便跑（实测跑到 5.8 米）；
    定得住的方向本来就是靠数据定的，这点罚相对残差可以忽略。
    """
    params = np.asarray(params, dtype=float)
    zeros = params[:len(ZERO_JOINTS)] / kinematics.MAX_ZERO_DEG
    offsets = params[len(ZERO_JOINTS):] / kinematics.MAX_OFFSET_M
    return REGULARIZATION * np.r_[zeros, offsets]


def board_poses(correction, model: ArmModel, joints, boards) -> np.ndarray:
    """每个位姿下板在基座系的位姿（4x4 一串），X 在里面顺手解出来。"""
    fk = model.forward(correction.get("offsets_m"))
    flanges = [fk(kinematics.corrected_joints(j, correction)) for j in joints]
    transform = handeye3d.solve(flanges, boards)
    return flanges, transform, [f @ transform @ b for f, b in zip(flanges, boards)]


def spread_mm(correction, model: ArmModel, joints, boards) -> tuple:
    """板位置散布（RMS/最大，毫米）与姿态散布（度）。这是"模型对不对"的唯一判据。"""
    _, _, poses = board_poses(correction, model, joints, boards)
    origins = np.array([pose[:3, 3] for pose in poses])
    errors = np.linalg.norm(origins - origins.mean(axis=0), axis=1) * 1000
    mean_rotation = handeye3d._mean_rotation([pose[:3, :3] for pose in poses])
    angles = np.array([handeye3d.rotation_angle_deg(mean_rotation.T @ pose[:3, :3]) for pose in poses])
    return float(np.sqrt((errors ** 2).mean())), float(errors.max()), float(np.sqrt((angles ** 2).mean()))


def _residual_vector(params, model: ArmModel, joints, boards) -> np.ndarray:
    """每个位姿的板位姿相对"全体均值"的偏差，(毫米, 毫弧度) 拼成一维。"""
    _, _, poses = board_poses(unpack(params), model, joints, boards)
    origins = np.array([pose[:3, 3] for pose in poses])
    mean_origin = origins.mean(axis=0)
    mean_rotation = handeye3d._mean_rotation([pose[:3, :3] for pose in poses])
    pieces = []
    for pose in poses:
        delta = pose[:3, 3] - mean_origin
        axis, angle = handeye3d.rotation_axis_angle(mean_rotation.T @ pose[:3, :3])
        rotvec = np.zeros(3) if axis is None else axis * angle
        pieces.append(np.r_[delta * 1000, rotvec * ROTATION_TO_MM])
    return np.r_[np.concatenate(pieces), _regularization(params)]


def identifiability(result) -> tuple[np.ndarray, int]:
    """每个参数的不确定度（1σ）和"这组数据实际能定住几个方向"。

    拟合给出的数好看，不代表每个数都被数据定住了——某些组合可能怎么变残差都一样
    （比如最后一个关节的零点：绕工具轴转一下，法兰位置不动、姿态变，会被 X 吸收掉）。
    用雅可比做 SVD：奇异值接近零的方向就是没定住的方向，报出来，别让人误以为都可信。
    """
    jacobian = result.jac
    dof = max(1, jacobian.shape[0] - jacobian.shape[1])
    variance = float((result.fun ** 2).sum() / dof)
    _, singulars, vt = np.linalg.svd(jacobian, full_matrices=False)
    # 奇异值近零的方向按零增益处理：那里不是"不确定度很大"，而是"数据压根没说"，
    # 报一个有限的大数反而会让人以为它有界。
    keep = singulars > (float(singulars.max()) * 1e-9 if singulars.size else 0.0)
    gain = np.where(keep, 1.0 / np.maximum(singulars, 1e-12) ** 2, 0.0)
    covariance = variance * (vt.T * gain) @ vt
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    # "定得住"的判据落在参数自己身上，而不是雅可比的秩：秩满不代表每个参数都被定住，
    # 有些方向是"几个参数一起动"才为零。1σ 小于物理上限的十分之一才算数。
    bounds = np.r_[np.full(len(ZERO_JOINTS), kinematics.MAX_ZERO_DEG),
                   np.full(3 * len(OFFSET_JOINTS), kinematics.MAX_OFFSET_M)]
    return sigma, int((sigma < bounds / 10.0).sum())


def fit(joints, boards, model: ArmModel, start=None) -> dict:
    """拟合几何修正。返回报告字典；不通过验证的也会如实写出来，由调用方决定用不用。

    Args:
        joints: N x 6 的关节角（度）。
        boards: N 个板在相机系的 4x4 位姿。
        model: ArmModel。
        start: 起点修正；默认全零。

    Returns:
        {"ok", "correction", "report", "before"/"after"（散布）, "heldout_before"/"heldout_after"}。
        不 ok 时 correction 仍会给出，但调用方不该落盘。
    """
    joints = np.asarray(joints, dtype=float)
    boards = list(boards)
    if len(joints) < MIN_POSES:
        return {"ok": False, "correction": kinematics.identity(),
                "report": "位姿只有 %d 个，少于 %d 个不拟合几何。" % (len(joints), MIN_POSES)}
    span = float(np.ptp(joints, axis=0).max())
    if span < MIN_SPREAD_DEG:
        return {"ok": False, "correction": kinematics.identity(),
                "report": "关节角跨度只有 %.1f°（要 ≥ %.0f°），这组数据定不住几何参数。"
                          "采集时让手腕多转一些。" % (span, MIN_SPREAD_DEG)}

    guess = pack(start) if start is not None else np.zeros(PARAM_COUNT)
    result = least_squares(_residual_vector, guess, args=(model, joints, boards),
                           diff_step=1e-4, max_nfev=200)
    correction = unpack(result.x)
    sigma, determined = identifiability(result)

    before = spread_mm(kinematics.identity(), model, joints, boards)
    after = spread_mm(correction, model, joints, boards)

    # 分半验证：隔一个取一个，两半轮流当拟合集和检验集。
    # 只看拟合集的话，参数够多总能压下去；要能在没见过的位姿上管用才算数。
    index = np.arange(len(joints))
    held = []
    for parity in (0, 1):
        fit_index = index[index % 2 == parity]
        test_index = index[index % 2 != parity]
        half = least_squares(_residual_vector, np.zeros_like(guess), args=(model, joints[fit_index],
                                                                          [boards[i] for i in fit_index]),
                             diff_step=1e-4, max_nfev=200)
        half_correction = unpack(half.x)
        plain = spread_mm(kinematics.identity(), model, joints[test_index],
                          [boards[i] for i in test_index])[0]
        fixed = spread_mm(half_correction, model, joints[test_index],
                          [boards[i] for i in test_index])[0]
        held.append((plain, fixed))
    heldout_before = float(np.mean([item[0] for item in held]))
    heldout_after = float(np.mean([item[1] for item in held]))
    improvement = 1.0 - heldout_after / heldout_before if heldout_before > 0 else 0.0

    good, why = kinematics.plausible(correction)
    ok = bool(good and improvement >= MIN_IMPROVEMENT)
    zeros = np.asarray(correction["zeros_deg"])
    offsets = np.asarray(correction["offsets_m"]) * 1000
    lines = [
        "臂几何自标定（%d 个位姿，关节角跨度 %.1f°）" % (len(joints), span),
        "  板位置散布：%.2f mm（最大 %.2f）→ %.2f mm（最大 %.2f）；姿态散布 %.3f° → %.3f°"
        % (before[0], before[1], after[0], after[1], before[2], after[2]),
        "  分半验证（隔一个取一个，拟合一半、检验另一半）：%.2f mm → %.2f mm（改善 %.0f%%）"
        % (heldout_before, heldout_after, improvement * 100),
        "  关节零点（度）：%s；1σ：%s"
        % (np.round(zeros, 3).tolist(), np.round(sigma[:len(ZERO_JOINTS)], 3).tolist()),
    ]
    for slot, joint in enumerate(OFFSET_JOINTS):
        lines.append("  J%d 的位置修正（mm）：%s；1σ：%s"
                     % (joint + 1, np.round(offsets[joint], 2).tolist(),
                        np.round(sigma[len(ZERO_JOINTS) + 3 * slot: len(ZERO_JOINTS) + 3 * slot + 3] * 1000, 2).tolist()))
    lines.append("  参数定得住的：%d / %d（判据：1σ 小于该参数物理上限的十分之一；"
                 "没定住的不代表它是零，只代表这批数据说不了它）"
                 % (determined, len(result.x)))
    if not good:
        lines.append("  结论：不采用——%s" % why)
    elif improvement < MIN_IMPROVEMENT:
        lines.append("  结论：不采用——分半验证只改善 %.0f%%，低于 %.0f%%，不值得改模型。"
                     % (improvement * 100, MIN_IMPROVEMENT * 100))
    else:
        lines.append("  结论：采用。这组修正会让由关节角算出来的法兰位姿更接近实物。")
    lines.append("  不参与的参数：J1 的平移（整条臂在基座系挪一下，和相机外参挪一下等价）、"
                 "J6 的零点与平移（整段落在法兰系里）——都会被手眼解出的 X 吸收，"
                 "留着只会让拟合在零空间里乱跑。")
    return {"ok": ok, "correction": correction, "report": "\n".join(lines),
            "before": before, "after": after,
            "heldout_before": heldout_before, "heldout_after": heldout_after,
            "improvement": improvement}
