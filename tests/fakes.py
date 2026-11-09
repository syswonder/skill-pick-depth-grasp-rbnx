# Description: 测试用的假对象。**只给 tests/ 用**，不进包、不连业务代码。
#
# 另一个并行队友写 pipeline 集成测试时可以 import 本文件共用这些假对象：
# **不要改 FakeCalibration**（它的几何被 observe_scan 的单测断言过），需要 client 这类
# 别的东西就在本文件末尾**新加类**。用的时候把 tests/ 放进 sys.path：
#     sys.path.insert(0, str(Path(__file__).resolve().parent))
#     import fakes
#
# ---------------------------------------------------------------------------
# 假 calib（FakeCalibration）的接口 —— observe_scan.plan_views 的鸭子类型：
#   .camera(joints_deg) -> 4x4        相机在基座系的位姿
#   .flange(joints_deg) -> 4x4        法兰在基座系的位姿（camera == flange @ X，与真 Calibration 一致）
#   .X                  -> 4x4        T_flange_camera：相机装在法兰前方 5 cm（无旋转）
#   .tip_in_flange      -> np[3]      爪尖相对法兰的偏移（米）
#   .bounds             -> [(lo,hi)]×6  弧度（真 Calibration 的形状；del 掉它测 joint_limits_deg 退路）
#   .joint_limits_deg() -> [(lo,hi)]×6  度
#   .observe_joints     -> [6]        标称拍照位（度）；集成测试可直接拿来当 nominal_joints_deg
#   .camera_matrix      -> 3×3        与 .image_size 配套的针孔内参
#   .image_size         -> (rows, cols) = (120, 160)
#
# 假 client（state()/gripper()/wrist_rgb()/wrist_depth()/move_joints…）不在这里：
# 那是 pipeline 那边集成测试自己的事，在文件末尾另写 FakeClient 就行。
# ---------------------------------------------------------------------------
import numpy as np


def make_transform(rotation=None, position=None) -> np.ndarray:
    """旋转（3×3，默认单位） + 平移（3，默认零）→ 4×4。"""
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    if position is not None:
        pose[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return pose


def camera_rotation(psi_rad, tilt_rad) -> np.ndarray:
    """假相机的姿态：R = Rz(ψ) · Ry(θ) · Rx(π)。

    光轴（第三列） = [-sinθ·cosψ, -sinθ·sinψ, -cosθ]——**朝下**，并且是朝基座竖轴那一侧
    倾斜 θ（θ>0 时落点往中心收，θ<0 往远处摊）。这套解析式子让单测能直接从关节角推出
    光轴/落点，不用 URDF 也不用真机标定。
    """
    cpsi, spsi = np.cos(psi_rad), np.sin(psi_rad)
    ctilt, stilt = np.cos(tilt_rad), np.sin(tilt_rad)
    return np.array([
        [cpsi * ctilt, spsi, -cpsi * stilt],
        [spsi * ctilt, -cpsi, -spsi * stilt],
        [-stilt, 0.0, -ctilt],
    ])


class FakeCalibration:
    """一台"解析几何"假臂：位姿全用手写公式算，不依赖 URDF/kinpy/真机标定。

    约定（基座系 z 朝上，地面/桌面在 z=surface_z < 0）：
      - q[0] = 绕基座 z 的方位角 ψ（度）；
      - q[1] = 相机相对"正俯视"的倾角 θ（度，正 = 往基座竖轴那一侧倾）；
      - q[2] = 相机位置半径的微调（1 mm/°）；
      - q[3] = 相机高度的微调（1 mm/°）；
      - q[4]、q[5] = **不改几何**，只用来验证"关节偏移量大了但画面没变"的候选会被
        旋转差/最远点采样筛掉。
    相机位置 p = [r·cosψ, r·sinψ, h]，其中 r = radius + 0.001·q[2]、h = height + 0.001·q[3]。
    """

    def __init__(self, radius_m: float = 0.30, height_m: float = 0.45,
                 tip_m: float = 0.20, camera_offset_m: float = 0.05,
                 image_size=(120, 160), focal_px: float = 300.0):
        self.base_radius_m = float(radius_m)
        self.base_height_m = float(height_m)
        # 相机在法兰前方 5 cm（纯平移），所以 flange 的旋转和 camera 一样，位置差 5 cm。
        self.X = make_transform(position=[0.0, 0.0, float(camera_offset_m)])
        self.tip_in_flange = np.array([0.0, 0.0, float(tip_m)])
        # 限位：比真 piper 宽一点，方便单测在它附近撒点而不被限位夹。
        self._limits_deg = [(-150.0, 150.0), (-100.0, 100.0), (-100.0, 100.0),
                            (-90.0, 90.0), (-90.0, 90.0), (-150.0, 150.0)]
        # 弧度那份：形状与真 Calibration.bounds 一致（URDF 原生弧度）。
        self.bounds = [(float(np.deg2rad(lo)), float(np.deg2rad(hi)))
                       for lo, hi in self._limits_deg]
        self.observe_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rows, cols = image_size
        self.image_size = (int(rows), int(cols))
        # 像素坐标与内参同一条约定（整数像素中心即坐标），和合成深度图那边一致。
        self.camera_matrix = np.array([
            [float(focal_px), 0.0, (cols - 1) / 2.0],
            [0.0, float(focal_px), (rows - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ])

    def camera(self, joints_deg) -> np.ndarray:
        """关节角 → 相机在基座系的位姿（解析式，见类说明）。"""
        joints = np.asarray(joints_deg, dtype=float).reshape(6)
        psi = np.deg2rad(joints[0])
        tilt = np.deg2rad(joints[1])
        radius = self.base_radius_m + 0.001 * joints[2]
        height = self.base_height_m + 0.001 * joints[3]
        position = [radius * np.cos(psi), radius * np.sin(psi), height]
        return make_transform(camera_rotation(psi, tilt), position)

    def flange(self, joints_deg) -> np.ndarray:
        """关节角 → 法兰在基座系的位姿。相机固定在法兰前方，所以 camera = flange @ X。"""
        return self.camera(joints_deg) @ np.linalg.inv(self.X)

    def joint_limits_deg(self) -> list:
        """六个关节的限位，[(lo, hi)]，度（plan_views 在没有 .bounds 时的退路）。"""
        return [(float(lo), float(hi)) for lo, hi in self._limits_deg]

    def set_limits_deg(self, limits_deg) -> None:
        """同时改 .bounds 和 .joint_limits_deg()（两边都由 _limits_deg 派生）。

        直接改 .bounds 或 ._limits_deg 只会改一边，plan_views 优先看 .bounds，
        测试就会测到旧限位——所以留这个方法，别手改。
        """
        self._limits_deg = [(float(lo), float(hi)) for lo, hi in limits_deg]
        self.bounds = [(float(np.deg2rad(lo)), float(np.deg2rad(hi)))
                       for lo, hi in self._limits_deg]


def synthetic_plane_depth(camera_pose, camera_matrix, surface_z, image_size=(120, 160),
                          noise_std_mm: float = 0.0, seed: int = 0,
                          mask_fingers: bool = True) -> np.ndarray:
    """合成一张"相机看着水平面 z=surface_z"的深度图（uint16，毫米）。

    每个像素从内参反出相机系射线，和平面求交；交点沿光轴的 z 就是那一格的深度值。
    画面最下面 20% 默认写成 0，模拟真机那帧里挡在下面的夹爪手指——estimate_surface_z
    应该绕开它，而不是把手指的高度当成地面。

    Args:
        camera_pose: 4×4，相机在基座系的位姿（cam → base）。
        camera_matrix: 3×3 内参（要与此处 image_size 匹配）。
        surface_z: 平面高度（基座系 z，米）。
        noise_std_mm: 深度噪声（毫米），0 = 理想平面。
        seed: 噪声种子（固定住，测试才确定性）。
        mask_fingers: 是否把画面底部 20% 置 0（模拟手指遮挡）。
    """
    rows, cols = image_size
    pose = np.asarray(camera_pose, dtype=float).reshape(4, 4)
    camera = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    fx, fy = float(camera[0, 0]), float(camera[1, 1])
    cx, cy = float(camera[0, 2]), float(camera[1, 2])
    u, v = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    directions = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    base_directions = directions @ pose[:3, :3].T
    downward = base_directions[..., 2] < -1e-9
    depth_mm = np.zeros((rows, cols), dtype=float)
    depth_mm[downward] = (surface_z - pose[2, 3]) / base_directions[..., 2][downward]
    valid = downward & (depth_mm > 0.0)
    depth_mm = np.where(valid, depth_mm * 1000.0, 0.0)
    if noise_std_mm:
        rng = np.random.default_rng(seed)
        depth_mm = np.where(valid, depth_mm + rng.normal(0.0, noise_std_mm, depth_mm.shape), 0.0)
    depth = np.clip(np.round(depth_mm), 0.0, 65535.0).astype(np.uint16)
    if mask_fingers:
        depth[int(rows * 0.8):, :] = 0
    return depth
