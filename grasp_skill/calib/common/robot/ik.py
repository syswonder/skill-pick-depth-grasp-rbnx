# Description: piper URDF 的运动学——解析运动学链与关节限位、数值求解逆运动学（阻尼最小二乘）。
#
# robot/piper_sdk.py（抓取用的臂后端）也用这里的解析结果，URDF 只在一处解析。
#
# 逆运动学的**核心**在 grasp_tilt/lm_ik.py（与仿真共用同一份实现）；这个文件只做
# "URDF 链这一侧"的适配：kinpy 的正运动学/雅可比、产物打包。为什么核心要共用，见那份的说明。
import re
import sys
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree

import kinpy
import numpy as np
from scipy.spatial.transform import Rotation as R

_GRASP_DIR = Path(__file__).resolve().parents[3] / "grasp_tilt"
if str(_GRASP_DIR) not in sys.path:
    sys.path.insert(0, str(_GRASP_DIR))
import lm_ik                                            # noqa: E402  与仿真共用的 LM 核心

# piper 的 URDF（随本子树放在 robot/urdf/piper/）
PIPER_URDF_PATH = Path(__file__).resolve().parent / "urdf" / "piper" / "piper_description.urdf"

# 末端连杆名（URDF 里最后一段）
END_LINK = "link6"


def build_chain_and_bounds(urdf_path=PIPER_URDF_PATH, correction=None):
    """解析 URDF。

    Args:
        urdf_path: URDF 路径。
        correction: 可选的臂几何修正；给了就按它的位置修正重建这条链，
            让 IK 算出来的关节角和"修正后的正运动学"是同一套模型（见 common/robot/kinematics.py）。

    Returns:
        (kinpy 串联链, 各 revolute 关节的 (lower, upper) 限位，弧度)。
    """
    content = Path(urdf_path).read_text(encoding="utf-8")
    # 去掉 XML 声明，否则 ElementTree 解析 unicode 字符串会报错
    content = re.sub(r"<\?xml[^?]*\?>", "", content, count=1)
    if correction is not None:
        from common.robot import kinematics

        if not kinematics.is_identity(correction):
            content = kinematics.corrected_content(
                content, END_LINK, np.asarray(correction.get("offsets_m", []), dtype=float))
    chain = kinpy.build_serial_chain_from_urdf(content, END_LINK)

    bounds: list[tuple[float, float]] = []
    for joint_elem in ElementTree.fromstring(content).findall("joint"):
        if joint_elem.get("type") != "revolute":
            continue
        limit = joint_elem.find("limit")
        if limit is None:
            print(f"警告：URDF 关节 {joint_elem.get('name')} 没有 limit，按不可动处理")
            bounds.append((0.0, 0.0))
            continue
        bounds.append((float(limit.get("lower", 0)), float(limit.get("upper", 0))))
    return chain, bounds


def arm_model():
    """按当前 URDF 建一个 ArmModel，给几何自标定用（它要按参数反复重建这条链）。"""
    from common import arm_geometry

    return arm_geometry.ArmModel(PIPER_URDF_PATH.read_text(encoding="utf-8"), END_LINK)


def joint_limits_deg(urdf_path=PIPER_URDF_PATH) -> list[tuple[float, float]]:
    """六个关节的角度限位，单位度。"""
    _, bounds = build_chain_and_bounds(urdf_path)
    return [(lo * 180.0 / np.pi, hi * 180.0 / np.pi) for lo, hi in bounds]


def clamp_joints_deg(joints_deg, margin_deg: float = 0.5,
                     urdf_path=PIPER_URDF_PATH) -> list[float]:
    """把目标关节角夹进限位，两边各留 margin_deg。

    顶到限位的关节，驱动器只能一直顶着，move_joints 到位后偏差过大直接抛异常。
    手眼采集里 J5 的观察角 69.99° 本身就贴着 ±69.90° 的限位，再往正方向抖就出界。
    """
    limits = joint_limits_deg(urdf_path)
    return [min(max(float(v), lo + margin_deg), hi - margin_deg)
            for v, (lo, hi) in zip(joints_deg, limits)]


class IKResult(NamedTuple):
    """口径与 scipy 的 OptimizeResult 一致（调用方只读 success / x / fun / message）。"""

    success: bool
    x: np.ndarray
    fun: float
    message: str


def _rotation_error(current, target) -> np.ndarray:
    """当前姿态到目标姿态的旋转向量（基座系）——与解析雅可比的角速度行同一套坐标。"""
    current = R.from_matrix(np.asarray(current, dtype=float))
    target = R.from_matrix(np.asarray(target, dtype=float))
    return current.apply((current.inv() * target).as_rotvec())


def _ik_error(chain, target, joints) -> np.ndarray:
    """关节角 → 到目标位姿的 6 维误差：[位置(米), 姿态(弧度)]，都在基座系。"""
    pose = np.asarray(chain.forward_kinematics(np.asarray(joints, dtype=float)).matrix(),
                      dtype=float)
    target = np.asarray(target, dtype=float)
    error = np.empty(6)
    error[:3] = target[:3, 3] - pose[:3, 3]
    error[3:] = _rotation_error(pose[:3, :3], target[:3, :3])
    return error


def solve_ik(pos, euler_zyx_deg, chain, bounds, n_starts: int = 15, seed: int = 42,
             start=None):
    """在多起点上跑 LM 求逆运动学解（核心见 grasp_tilt/lm_ik.py，与仿真**共用同一份**）。

    这里只干"URDF 链这一侧"的事：把 kinpy 的正运动学/雅可比包成核心要的两个回调、把
    结果包成旧签名认的 IKResult。算法、种子铺法、判据全在 lm_ik 里。

    Args:
        pos: 目标末端（法兰）位置 [x, y, z]，基座系，米。
        euler_zyx_deg: 目标末端欧拉角 [RZ, RY, RX]，度。
        chain / bounds: build_chain_and_bounds 的产物（链、关节限位，弧度）。
        n_starts: 最多从几个起点下降。**第一颗优先用 start**（臂现在的关节角）：它到位
            就收手——既是"尽量别换分支"，也省掉其余尝试（实测：第一颗到位时，一次求解
            476 ms → 2.2 ms）。
        seed: 只为兼容旧签名留着；起点已是确定性铺开，它不参与求解。
        start: 可选，当前关节角（弧度）。夹着方块走的那些调用要传它。

    Returns:
        IKResult：`x` 为关节角（弧度）、`fun` 为加权代价（位置米 + 0.3×姿态弧度）。
        **success 只表示"有解交回"，不表示"到位"**——到没到位看 message，或者调用方
        自己按 FK 量出来的误差判（与仿真那份同一个口径）。
    """
    target = kinpy.Transform(
        pos=np.array(pos, dtype=float),
        rot=R.from_euler("zyx", euler_zyx_deg, degrees=True).as_euler("xyz"),
    ).matrix()
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    result = lm_ik.multi_start(
        lambda joints: _ik_error(chain, target, joints),
        lambda joints: np.asarray(chain.jacobian(joints), dtype=float).reshape(6, len(lo)),
        lo, hi, start=start, budget=max(1, int(n_starts)))
    error = result["error"]
    position_error = float(np.linalg.norm(error[:3]))
    rotation_error = float(np.linalg.norm(error[3:]))
    if lm_ik.converged(error):
        message = ("LM 收敛：第 %d 颗起点，位置 %.2f mm、姿态 %.2f°"
                   % (result["attempts"][-1]["index"], position_error * 1000,
                      np.degrees(rotation_error)))
    else:
        message = ("LM 没到位：最小位置误差 %.2f mm、姿态 %.2f°，交回最接近的解"
                   % (position_error * 1000, np.degrees(rotation_error)))
    return IKResult(True, result["joints"], lm_ik.cost(error), message)
