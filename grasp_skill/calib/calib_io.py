# Description: 读标定产物、给 IK 链——这个目录里唯一一处依赖标定工具的地方。
#
# 标定工具（tools/calibration）里的眼在手上标定台产出全套参数：手眼外参 X、相机内参、
# 臂几何修正、爪尖偏移、夹爪抓取尺寸。抓取要用的是这些数，不是那套页面，所以这里只读
# artifacts 下的文件，不导入标定台的应用代码。
#
# 唯一借用的是运动学那部分（URDF + 修正重建的链、数值 IK）：它按修正把 URDF 改好再建链，
# 是标定那侧算过的同一套模型，重写一份只会有两份会漂的模型。将来打包时，这一段换成
# 臂原语自己的 IK 接口，其余文件不受影响。
import json
import sys
from pathlib import Path

import numpy as np

# 技能包里这些依赖跟着包走：paths.py / pose.py 就在本目录，common/robot/ 在本目录下。
# 所以只加两条路径，不再去找标定工具那棵树。
CALIB_ROOT = Path(__file__).resolve().parent
for _p in (str(CALIB_ROOT), str(CALIB_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paths as calib_paths                             # noqa: E402  标定产物路径
from common.robot import kinematics                     # noqa: E402
from common.robot.ik import build_chain_and_bounds, solve_ik  # noqa: E402
from pose import flange_of                              # noqa: E402

# 缺这些就没有可用的抓取参数，早点报出来比走到一半再炸强。
# **相机内参/畸变系数不在这里**：视觉算法（含取帧、检测、平面/顶面、"框→三维"）
# 已经搬到 service/perception/grasp_geometry，内参是服务自己的配置；技能只做
# "相机系→基座系"那一乘（手眼外参 X）和 IK（爪尖偏移）。旧部署的标定产物里
# 仍然会有内参文件，有就顺手读、没有也不影响技能。
REQUIRED = {
    "手眼外参 X": calib_paths.GRIPPER_CAMERA,
    "爪尖偏移": calib_paths.TIP_IN_FLANGE,
}

# 不读文件那一路（from_numbers）必须有的键，语义与 REQUIRED 一一对应
REQUIRED_NUMBERS = ("gripper_camera", "tip_in_flange")

# "到位"的判据（米）：位置误差 + 0.3×姿态误差（弧度）不超过它才算到位。与 sim_ik 同一套。
SETTLED_M = 0.0005 + 0.3 * 0.005


class Calibration:
    """一份标定参数，外加用它建出来的 IK 链。

    链只建一次：解析 URDF 加重建要几十毫秒，每次求解都建会拖慢规划。

    两份来源，内容一样，用哪个由部署决定：
      - `Calibration()`    读标定产物目录（开发台、标定台旁边用）
      - `from_numbers()`   直接用写在部署清单里的数（机器人上的容器用：那边
                           `tools/` 不在镜像里，标定产物目录是拿不到的）
    """

    def __init__(self):
        missing = [name for name, path in REQUIRED.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("标定产物不全：%s。先到标定台把这几步走完。"
                                    % "、".join(missing))
        numbers = {
            "correction": kinematics.load(calib_paths.KINEMATICS_CORRECTION),
            "gripper_camera": np.load(calib_paths.GRIPPER_CAMERA),     # T_gripper_camera
            "tip_in_flange": np.load(calib_paths.TIP_IN_FLANGE),
            "observe_joints_deg": (json.loads(
                calib_paths.OBSERVE_POSE.read_text(encoding="utf-8"))["joints_deg"]
                if calib_paths.OBSERVE_POSE.exists() else None),
            "gripper_size": self._load_gripper(),
        }
        # 内参是视觉服务那边的事，技能不需要；旧标定产物里有就读进来（诊断/兼容用）。
        if calib_paths.CAMERA_MATRIX.exists():
            numbers["camera_matrix"] = np.load(calib_paths.CAMERA_MATRIX)
        if calib_paths.DIST_COEFFS.exists():
            numbers["dist_coeffs"] = np.load(calib_paths.DIST_COEFFS)
        self._use(numbers)

    @classmethod
    def from_numbers(cls, numbers: dict) -> "Calibration":
        """不碰文件，直接用清单 config 里那组数建一份。

        数从哪来：标定台的产物用 `tools/calibration/eyeinhand/dump_config.py` 印成
        config 片段，贴进部署清单。**数一旦挪进清单就不再跟着标定台走了**——重标之后
        要重新导一次，否则用的是旧标定。这一点和"读产物目录"不一样，那边永远是新的。
        """
        obj = cls.__new__(cls)
        obj._use(dict(numbers or {}))
        return obj

    def _use(self, numbers: dict) -> None:
        """两条路汇到这里：摆好属性、建一次链。"""
        missing = [name for name in REQUIRED_NUMBERS if numbers.get(name) is None]
        if missing:
            raise ValueError("标定参数不全：%s" % "、".join(missing))
        correction = numbers.get("correction")
        if correction:
            ok, why = kinematics.plausible(correction)
            if not ok:
                raise ValueError("运动学修正不可信：%s" % why)
        self.correction = correction
        self.chain, self.bounds = build_chain_and_bounds(correction=self.correction)
        self.X = np.asarray(numbers["gripper_camera"], dtype=float).reshape(4, 4)
        # 相机内参/畸变系数**不再是技能的必需项**（视觉在服务里）；给了就留着，
        # 兼容旧部署与离线测试对 `calib.camera_matrix` 的引用。
        self.camera_matrix = (None if numbers.get("camera_matrix") is None
                              else np.asarray(numbers["camera_matrix"],
                                              dtype=float).reshape(3, 3))
        self.dist_coeffs = (None if numbers.get("dist_coeffs") is None
                            else np.asarray(numbers["dist_coeffs"],
                                            dtype=float).reshape(-1, 1))
        self.tip_in_flange = np.asarray(numbers["tip_in_flange"], dtype=float).reshape(3)
        self.observe_joints = numbers.get("observe_joints_deg")
        self.gripper = numbers.get("gripper_size")
        # 臂现在在哪儿（由 seed() 更新）：带闸求解与等价解挑选的基准
        self._last_joints_deg: list | None = None
        # 上一次 solve 的内部过程（哪几支、各自的误差与关节移动、最后选了谁）。
        # 上层把它写进日志（pipeline 的 descend_ik）——见 _solve_debug 的说明。
        self.last_solve_debug: dict | None = None

    @staticmethod
    def _load_gripper() -> dict | None:
        path = calib_paths.GRIPPER_SIZE
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # ---------- 位姿 ----------

    def flange(self, joints_deg) -> np.ndarray:
        """关节角 → 法兰在基座系的位姿（带几何修正）。"""
        return flange_of(list(joints_deg), None, None, self.correction)

    def flange_from_feedback(self, feedback: dict) -> np.ndarray:
        """臂上报的关节角与位姿 → 法兰位姿；有修正时以关节角为准。"""
        return flange_of(feedback["joints_deg"], feedback["pos"], feedback["euler_zyx"],
                         self.correction)

    def camera(self, joints_deg) -> np.ndarray:
        """关节角 → 相机在基座系的位姿。"""
        return self.flange(joints_deg) @ self.X

    def tip(self, flange_pose) -> np.ndarray:
        """法兰位姿 → 爪尖在基座系的位置。"""
        pose = np.asarray(flange_pose, dtype=float)
        return pose[:3, 3] + pose[:3, :3] @ self.tip_in_flange

    # ---------- IK ----------

    def seed(self, joints_deg) -> None:
        """记下"臂现在在哪儿"：下一次 solve 拿它当基准（关节移动闸、等价解挑选都要它）。

        与仿真那份（sim/sim_ik.py）同一套语义——pipeline 里凡是夹着方块、贴着物体走的
        步骤都会先 seed 再解，靠它判断"这一步要动多少度"。
        """
        self._last_joints_deg = [float(v) for v in joints_deg]

    def joint_limits_deg(self) -> list:
        """六个关节的角度限位 [(lo, hi)]，度（回位挑"腕子转得少"的等价角要用）。"""
        return [(float(np.rad2deg(lo)), float(np.rad2deg(hi))) for lo, hi in self.bounds]

    def solve(self, tip_pos, rotation, n_starts: int = 15, seed: int = 42,
              max_joint_move_deg: float | None = None,
              settled_m: float | None = None):
        """求一条能到位且姿态对的解。

        Args:
            tip_pos: 爪尖目标位置（基座系，米）。
            rotation: 目标姿态 3x3。爪尖偏移沿这个姿态从法兰推出来。
            n_starts / seed: 透传给数值 IK。
            settled_m: "算到位"的判据（米）。不给就用下面的 SETTLED_M（2 mm）。**下爪那一步
                要放宽**：它是"从悬停往物体上落"，能用的是同一副臂，收得比 2 mm 紧的结果
                是"差 0.3 mm 就一步不走"——而差几毫米照样夹得住（见 pipeline.DESCEND_SETTLED_M）。
            max_joint_move_deg: 夹着方块/贴着物体走的时候传它——只有"到位 **且** 关节相对
                `seed()` 记下的姿态没走超过这个角度"的解才算数。一个都不算数时**原地不动**、
                如实交回当前姿态与真实误差；这条与仿真那份是同一套判据，仿真里正是它挡住了
                "整条臂横扫方块"，真机上更不能没有。夹爪绕工具轴 180° 对称（同一个夹法有
                两族解），主解不算数时会绕工具轴翻 180° 再解一次，挑转得少的那族。

        Returns:
            {"ok", "joints_deg", "flange", "tip", "position_error_m", "tilt_error_deg",
             "reached_rotation", "message"}。ok 只表示 IK 收敛且姿态可复现，
             能不能真的走过去由调用方按误差阈值判断。
        """
        solved = self._solve_target(tip_pos, rotation, n_starts, seed)
        keep = getattr(self, "_last_joints_deg", None)
        settled = SETTLED_M if settled_m is None else float(settled_m)
        if keep is None or max_joint_move_deg is None:
            # 没闸（或没 seed 过）就不筛，主解直接算数
            self.last_solve_debug = _solve_debug(
                [("primary", solved, None, True)], "primary",
                max_joint_move_deg, keep, tip_pos, rotation, settled)
            return solved
        move_limit = float(np.deg2rad(max_joint_move_deg))
        move = _move_deg(solved, keep)                 # 度（_move_deg 的输出就是度）
        primary_ok = _acceptable(solved, move, move_limit, settled)
        attempts = [("primary", solved, move, primary_ok)]
        if primary_ok:
            self.last_solve_debug = _solve_debug(attempts, "primary", max_joint_move_deg,
                                                 keep, tip_pos, rotation, settled)
            return {**solved, "joint_move_deg": float(move)}
        rolled = np.asarray(rotation, dtype=float).reshape(3, 3) @ np.diag([-1.0, -1.0, 1.0])
        twin = self._solve_target(tip_pos, rolled, n_starts, seed)
        twin_move = _move_deg(twin, keep)
        twin_ok = twin_move < move and _acceptable(twin, twin_move, move_limit, settled)
        attempts.append(("twin", twin, twin_move, twin_ok))
        if twin_ok:
            self.last_solve_debug = _solve_debug(attempts, "twin", max_joint_move_deg,
                                                 keep, tip_pos, rotation, settled)
            return {**twin, "joint_move_deg": float(twin_move)}
        # 一支都不算数：原地不动，把真实误差交回去（调用方据此报"没到位"，不许扫）
        self.last_solve_debug = _solve_debug(attempts, "keep_fallback", max_joint_move_deg,
                                             keep, tip_pos, rotation, settled)
        pose = self.flange(keep)
        reached_tip = self.tip(pose)
        error = float(np.linalg.norm(reached_tip - np.asarray(tip_pos, dtype=float)))
        return {
            "ok": True, "joints_deg": [float(v) for v in keep], "flange": pose.tolist(),
            "tip": reached_tip.tolist(), "rotation": pose[:3, :3].tolist(),
            "target_tip": np.asarray(tip_pos, dtype=float).tolist(),
            "position_error_m": error, "tilt_error_deg": 0.0, "joint_move_deg": 0.0,
            "message": "没有解能在关节移动闸（%.0f°）内到位，原地不动" % max_joint_move_deg,
        }

    def _solve_target(self, tip_pos, rotation, n_starts: int, seed: int) -> dict:
        """真解一次（不带闸）：原 solve 的主体。"""
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        target_flange_pos = np.asarray(tip_pos, dtype=float) - rotation @ self.tip_in_flange
        euler_zyx_deg = _euler_zyx_deg(rotation)
        # 第一颗起点用"臂现在在哪儿"（seed() 记下的那份）：LM 从当前姿态往下走，既快
        # （到位时第一颗就收手）又不会无缘无故换分支。没 seed 过（比如第一次求解）就
        # 用铺开的确定性起点。
        start = (None if getattr(self, "_last_joints_deg", None) is None
                 else np.deg2rad(np.asarray(self._last_joints_deg, dtype=float)))
        result = solve_ik(target_flange_pos, euler_zyx_deg, self.chain, self.bounds,
                          n_starts=n_starts, seed=seed, start=start)
        if result is None or not result.success:
            return {"ok": False, "message": "IK 没收敛：%s" % (None if result is None else result.message)}
        reached = np.asarray(self.chain.forward_kinematics(result.x).matrix())
        reached_tip = reached[:3, 3] + reached[:3, :3] @ self.tip_in_flange
        return {
            "ok": True,
            "joints_deg": [float(v) for v in np.rad2deg(result.x)],
            "flange": reached.tolist(),
            "tip": reached_tip.tolist(),
            "rotation": reached[:3, :3].tolist(),          # 实际到位的姿态，偏差要按它算
            "target_tip": np.asarray(tip_pos, dtype=float).tolist(),
            "position_error_m": float(np.linalg.norm(reached_tip - np.asarray(tip_pos, dtype=float))),
            "tilt_error_deg": float(_rotation_gap_deg(reached[:3, :3], rotation)),
            "message": str(getattr(result, "message", "")),
        }


def _solve_debug(attempts, selected, move_limit_deg, keep, target_tip, target_rotation,
                 settled=None) -> dict:
    """把这一次 solve 的内部过程记下来，供上层写进日志（pipeline 里的 descend_ik）。

    为什么要有它：真机上"下爪没到位"只报一个毫米数，看不出是**根本没解出来**还是
    **解出来了但超出关节移动闸**——这两者的原因和改法完全不同。仿真那份一直记着
    （sim/sim_ik.py 的 last_solve_debug），真机这份之前没这个属性，于是上层的
    descend_ik 恒为 null，而那个字段本来就是为这件事留的。

    Args:
        attempts: [(分支名, 解, 关节移动度数, 过没过判据)]，按尝试顺序。
        selected: 最后采用的分支（"primary" / "twin" / "keep_fallback"）。
    """
    return {
        "attempts": [{
            "branch": branch,
            "ok": bool(solved.get("ok")),
            "position_error_m": solved.get("position_error_m"),
            "tilt_error_deg": solved.get("tilt_error_deg"),
            "joint_move_deg": (None if move is None or not np.isfinite(move)
                               else float(move)),
            "acceptable": bool(acceptable),
            "message": solved.get("message", ""),
        } for branch, solved, move, acceptable in attempts],
        "selected": selected,
        "move_limit_deg": move_limit_deg,
        # 实际用的那条判据——不是模块常量：下爪那一步是按爪和物体的尺寸现算的
        # （见 grasp.descend_settled_m）。以前这里印 SETTLED_M，日志里看到的判据
        # 跟真正用的不是一回事。
        "settled_m": SETTLED_M if settled is None else float(settled),
        "keep_joints_deg": None if keep is None else [float(v) for v in keep],
        "target_tip": np.asarray(target_tip, dtype=float).tolist(),
        "target_rotation": np.asarray(target_rotation, dtype=float).reshape(3, 3).tolist(),
    }


def _move_deg(solved: dict, keep) -> float:
    """这个解相对 keep 最大动了几度；没解出来算无穷大（必被闸挡下）。"""
    if not solved.get("ok") or solved.get("joints_deg") is None:
        return float("inf")
    return float(np.max(np.abs(np.asarray(solved["joints_deg"], dtype=float)
                               - np.asarray(keep, dtype=float))))


def _acceptable(solved: dict, move_deg: float, move_limit_rad: float, settled: float) -> bool:
    """到位 + 没超出关节移动闸，两个条件一起看（判据同 sim_ik）。

    `move_deg` 是度（`_move_deg` 的输出），`move_limit_rad` 是弧度（`np.deg2rad(闸角度)`）。
    """
    if not solved.get("ok"):
        return False
    error = solved["position_error_m"] + 0.3 * np.deg2rad(solved["tilt_error_deg"])
    return error <= settled and np.deg2rad(move_deg) <= move_limit_rad


def _euler_zyx_deg(rotation) -> list:
    """3x3 → 臂接口要的 [RZ, RY, RX] 欧拉角（度）。"""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_euler("zyx", degrees=True).tolist()


def _rotation_gap_deg(a, b) -> float:
    """两个旋转之间的夹角，度。"""
    cosine = (np.trace(np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))
