# Description: 三维标定产物的路径，全子树唯一来源。
#
# 产物是这套工具与部署之间的接口，标定完要抄进部署清单（deploy/s10/manifest.yaml）：
#   intrinsics/{camera_matrix,dist_coeffs}.npy → 清单里的相机内参与畸变系数
#   handeye/gripper_camera.npy                 → 相机装在夹爪上的位姿 X（4x4）
#   handeye/board_in_base.npy                  → 标定板在基座系的位姿，顺带给出桌面高度
#   kinematics/correction.json                 → 臂的几何修正（关节零点 + 各关节位置修正）；
#                                                由关节角算法兰位姿的地方都该带上它
#   tcp/tip_in_flange.npy                      → 夹爪尖相对法兰的位置（3 个数，米）
#   board.json                                 → 标定板规格（格子数、边长、码表）
#   poses/observe_pose.json                    → 观察位姿的关节角（第一步定下来的）
#   poses/taught_pose.json                     → 示教记录，多条带时间戳
#
# 与平面标定（s10/paths.py）的区别：那边产出 2d_homography.npy，只能算贴在桌面上的物体；
# 这边产出 gripper_camera.npy，深度相机看到的任意三维点都能换到基座系。
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 产物根目录。标定工具那边是 ./artifacts；技能包里产物不随包走（在标定工具那棵树下），
# 所以由调用方在 import 之前用 GRASP_DEV_CALIB_ROOT 指定。下面这些常量是 import 时
# 算出来的，所以只能用环境变量，不能等 import 完再设。
ARTIFACTS = Path(os.environ.get("GRASP_DEV_CALIB_ROOT") or (ROOT / "artifacts"))

# 相机内参（手眼那步从同一批照片标出来，见 vision/intrinsics.py）
INTRINSICS_DIR = ARTIFACTS / "intrinsics"
CAMERA_MATRIX = INTRINSICS_DIR / "camera_matrix.npy"
DIST_COEFFS = INTRINSICS_DIR / "dist_coeffs.npy"
FACTORY_INTRINSICS = INTRINSICS_DIR / "factory.json"      # 相机自己报的内参，用来核对
INTRINSICS_REPORT = INTRINSICS_DIR / "report.txt"

# 标定板规格（标定台「标定板」那页产出）
BOARD_SPEC = ARTIFACTS / "board.json"

# 手眼标定（自动手眼）
HANDEYE_DIR = ARTIFACTS / "handeye"
GRIPPER_CAMERA = HANDEYE_DIR / "gripper_camera.npy"       # X = T_gripper_camera，4x4
BOARD_IN_BASE = HANDEYE_DIR / "board_in_base.npy"         # 标定板在基座系的位姿，4x4
HANDEYE_SAMPLES = HANDEYE_DIR / "samples.json"            # 兼容旧格式；自动手眼按 round_*_samples.json 保存
HANDEYE_IMAGES = HANDEYE_DIR / "images"
HANDEYE_REPORT = HANDEYE_DIR / "report.txt"

# 臂几何修正（自动手眼那页从多机位数据里拟合出来的关节零点与位置修正）
# 没有这个文件时，工具全部按标称 URDF 走，行为和以前一样。
KINEMATICS_DIR = ARTIFACTS / "kinematics"
KINEMATICS_CORRECTION = KINEMATICS_DIR / "correction.json"
KINEMATICS_REPORT = KINEMATICS_DIR / "report.txt"

# 夹爪尖标定（标定台「夹爪尖」那页产出）
TCP_DIR = ARTIFACTS / "tcp"
TIP_IN_FLANGE = TCP_DIR / "tip_in_flange.npy"             # 夹爪尖在法兰系的位置，3 个数
TCP_SAMPLES = TCP_DIR / "samples.json"                    # 每次碰点的关节角与法兰位姿
TCP_REPORT = TCP_DIR / "report.txt"
# 抓取几何（「夹爪」那页人工量出来）：可用夹持段长度、最大开口、合上间隙。
# 用来算下爪深度——爪尖停多高，物体才落在两指之间；手眼那条链不用这几个数。
GRIPPER_SIZE = TCP_DIR / "gripper_size.json"

# 位姿（标定台「定相机位置」那页产出）
POSES_DIR = ARTIFACTS / "poses"
CAMERA_DIR = ARTIFACTS / "camera"
CAMERA_SETTINGS = CAMERA_DIR / "settings.json"            # 曝光/增益/激光功率（「相机」页调的）
OBSERVE_POSE = POSES_DIR / "observe_pose.json"            # 观察位姿的关节角
TAUGHT_POSE = POSES_DIR / "taught_pose.json"              # 示教记录，多条带时间戳

# 验证（标定台「验证」那页产出）
VERIFY_REPORT = ARTIFACTS / "verify_report.txt"
# 实测落点误差：每点一次让臂走过去，人拿尺子量爪尖离目标多远，记一行。
# 全部是算出来的数里，只有这一列是量出来的。
VERIFY_MEASUREMENTS = ARTIFACTS / "verify_measurements.jsonl"

# 平面标定那边也有一份 HOMOGRAPHY，common/robot/arm.py 会引用；这套工具不产它，
# 指到手眼目录下让引用不至于报错，抓取走的是 GRIPPER_CAMERA。
HOMOGRAPHY = HANDEYE_DIR / "2d_homography.npy"
