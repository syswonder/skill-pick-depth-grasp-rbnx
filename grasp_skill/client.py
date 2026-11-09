# Description: 抓取的话题客户端（**真机这一份**）。状态与命令都走 ROS 话题。
#
# 话题名一律由编排层经 atlas 解析 `robonix/primitive/arm/*` 契约拿到真名后注入
# （见 atlas_bridge）——不是写死的、也不猜。
#
# **本客户端不订阅相机**：视觉（含取帧）整个在 service/perception/grasp_geometry
# 那一侧，技能只经 `vision_service` 调它的 locate。这里保留的是臂的状态/命令与
# 由标定链算出来的相机位姿（`state()["camera_pose"]`，给相机系→基座系那一步用）。
#
# 这里**只有真机**：没有 /sim/objects 那套物体位姿（本包不订阅它，真机上也没有这个
# 数据源）。仿真那份在 tools/grasp_tilt/client.py，两边各自推进。
#
# **状态只从话题来**：本地模型只许出现在 IK 求解里，绝不作为状态的来源。
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import JointState

import motion

# 话题名。上 robonix 时由编排层在激活阶段经 atlas 解析 `robonix/primitive/arm/*`
# 契约拿到真名，再从这里注入——**不猜**：解析不到且没有默认值的项直接不订阅，
# 让动作时报错，而不是挂在一个猜出来的话题上静默失联。
DEFAULT_TOPICS = {
    "joints": "/arm/joint_states_single",
    "command": "/arm/joint_command",
    "end_pose": "/arm/end_pose",
}
JOINTS_TOPIC = DEFAULT_TOPICS["joints"]
COMMAND_TOPIC = DEFAULT_TOPICS["command"]
END_POSE_TOPIC = DEFAULT_TOPICS["end_pose"]
ARM_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# 夹爪命令是单指行程（米），反馈是开口宽度（米）：话题两端口径不同，转换只在这里做。
SETTLE_TOL_DEG = 1.0
# 合爪命令发了之后，实际开口比命令值宽出这么多，就算指间有东西（见 holding）
HOLDING_TOL_M = 0.0005
SETTLE_TIMEOUT_S = 12.0
# 关节运动的插值步长与步间间隔。不插值的话位置伺服一步到位（实测 0.15 秒走完一大段，
# 等效 400 度/秒以上），比真机快 5~10 倍，看着像闪现。
#
# 标称 3 度 / 30 毫秒 = 100 度/秒，但**实测只跑到约 19 度/秒**：伺服跟不上插值的步子，
# 每步大约只走到命令量的五分之一（J2 转 68 度实测要 3 秒多）。想要更快就调大
# MOVE_STEP_DEG，想更慢就调小——两个数一起定速度，改完按实测校准。
MOVE_STEP_DEG = 3.0
MOVE_STEP_S = 0.03
GRIPPER_MOVE_M = 0.0002      # 反馈比开始时差这么多才算"真的动了"
                             # （合爪小步 0.5 mm，阈值不能比步长还大，否则每步都判成没动）
GRIPPER_STABLE_M = 0.0005    # 相邻采样差这么多以内算"没在动"
GRIPPER_STABLE_S = 1.0       # 动过之后连续这么久不动才算停住。
                             # 夹爪的伺服欠阻尼（ζ≈0.06），到位前会以 ~0.1 s 的周期
                             # 振荡；0.1 s 采样会停在振荡的同一相位上，把振荡误判成
                             # "停住了"（实测：接触检测提前 2.3 mm 误报）。采样改密、
                             # 判停窗口拉长到 1 s，振荡衰减完了才认。
GRIPPER_SAMPLE_S = 0.02
# 合爪最后一下的"真停住"判据：一个窗口内极差小于这么多才算停。不能用 GRIPPER_STABLE_M
# （相邻采样差 0.5 mm 就算没动）——指头慢慢爬的时候每次都小于它，会被当成停住，于是
# 合爪那一步在还没闭上时就结束，抬起途中指头才夹到方块（现场：下爪时夹具没闭拢、
# 边抬边闭、夹到方块就滑脱）。
GRIPPER_SETTLE_WINDOW_S = 1.5
GRIPPER_SETTLE_RANGE_M = 0.00005
# 数据源超过这么久没更新，就认定订阅断了（见 stale_sources）。剩下的两个数据源
# （关节 100 Hz、末端位姿约 10 Hz）都远快于它。
STALE_AFTER_S = 4.0


def _pose_from(position, quat_xyzw) -> np.ndarray:
    """位置 + 四元数（xyzw）→ 4x4 位姿。"""
    x, y, z, w = [float(v) for v in quat_xyzw]
    matrix = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    pose = np.eye(4)
    pose[:3, :3] = matrix
    pose[:3, 3] = np.asarray(position, dtype=float)
    return pose


class Client(Node):
    """抓取要用的话题客户端。所有读取都带超时：话题断了就报错，不拿旧值干下去。"""

    def __init__(self, calib, node_name: str = "grasp_tilt_client", topics: dict | None = None):
        super().__init__(node_name)
        # 空的项（"")表示"这个数据源本部署没有"，直接不订阅
        self.topics = {**DEFAULT_TOPICS, **(topics or {})}
        self.calib = calib
        self._joints = None               # {"joints_deg": (6,), "gripper_m": 开口宽度, "stamp": 秒}
        self._end_pose = None             # 4x4 世界系（话题原值）。**只做对照**：法兰位姿
                                          # 一律用标定链从关节角算，见 state()
        self._gripper_cmd = 0.0           # 夹爪命令值（单指行程，米），每次发命令都带上
        # 每个数据源最后一次收到的时刻。用**单调钟**，不用消息里的仿真时间——后者
        # 会被页面上的 Reset 归零，拿它判新鲜度会误判。
        self._seen = {}
        self._subscriptions = []
        self.publisher = None
        self._subscribe_all()

    # ---------- 链路 ----------

    # 数据源名 → (话题、消息类型、回调、队列深) 的表。话题为空就跳过这一项。
    def _subscribe_all(self) -> None:
        # 话题键, 消息类型, 回调, QoS, **落进 _seen 的名字**。（相机不在这里：
        # 帧由视觉服务自己取，技能端不订阅任何图像话题。）
        wanted = (
            ("joints", JointState, self._on_joints, 20, "joints"),
            ("end_pose", Pose, self._on_end_pose, 20, "end_pose"),
        )
        self._subscriptions = [
            self.create_subscription(kind, self.topics[key], callback, depth)
            for key, kind, callback, depth, _ in wanted if self.topics.get(key)
        ]
        # 只把这几个当"该来数据"的源：没订的源不该被等。名单从 `wanted` 里现推，
        # 不写死——写死过一个真机上根本不订阅的源，结果 `stale_sources()` 永远非空，
        # 一调就死在那道检查上（见 stale_sources）。
        self._watched = [seen for key, _, _, _, seen in wanted if self.topics.get(key)]
        if self.publisher is None and self.topics.get("command"):
            self.publisher = self.create_publisher(JointState, self.topics["command"], 20)

    def holding(self) -> bool | None:
        """还握着东西吗——按"指头没合到命令位"判断。

        真机上"物体升起来了没有"这条判据等于没有（client 里的物体位置是拍照位那一眼估的
        旧值，物体被夹起来它不会变）。真机给得
        出的只有这一条：合爪命令压下去之后指头停住了，说明指间有东西。

        仿真那份 client 没有这个方法，流水线用 getattr 取，取不到就退回物体位姿判据。
        """
        if self._joints is None:
            return None                       # 还没有关节反馈，别猜
        return abs(self._joints["gripper_m"] - self._gripper_cmd) > HOLDING_TOL_M

    def stale_sources(self, max_age_s: float = STALE_AFTER_S) -> list:
        """超过 max_age_s 没更新的数据源。空列表＝都在正常来。

        为什么要这个：运行时的桥重启过之后，**这个节点上的订阅会静默失效**——话题
        还在发（新起的订阅者收得到），这个节点却收不到了，于是后面全拿冻住的旧数据
        算。现象很像"算法坏了"（方块位置是旧的、目标点算错、IK 解不出来），实际是
        数据源早断了。实测撞过三次，每次只能重起后端。

        名单来自 `_subscribe_all` 记下的 `_watched`，**只含真订了的源**——不能写死。
        曾经写死过一个真机上压根不订阅的源，它 `_seen` 永远不写、于是永远"不新鲜"，
        `stale_sources()` 永远非空，一调就死在那道检查上，还报成"话题那头没在发"。
        """
        now = time.monotonic()
        return [name for name in self._watched
                if now - self._seen.get(name, 0.0) > max_age_s]

    def reset_links(self) -> None:
        """把订阅拆了重建。配合 stale_sources 用，见那里的说明。"""
        for subscription in self._subscriptions:
            self.destroy_subscription(subscription)
        self._subscriptions = []
        self._subscribe_all()

    # ---------- 话题回调 ----------

    def _on_joints(self, message: JointState) -> None:
        values = dict(zip(message.name, message.position))
        if not all(name in values for name in ARM_NAMES):
            return
        self._seen["joints"] = time.monotonic()
        stamp = _stamp_seconds(message.header.stamp)
        self._joints = {
            "joints_deg": np.array([values[name] for name in ARM_NAMES]) * 180.0 / np.pi,
            "gripper_m": float(values.get("gripper", 0.0)),
            "stamp": stamp,
        }

    def _on_end_pose(self, message: Pose) -> None:
        self._seen["end_pose"] = time.monotonic()
        quaternion = message.orientation
        self._end_pose = _pose_from([message.position.x, message.position.y,
                                     message.position.z],
                                    [quaternion.x, quaternion.y, quaternion.z,
                                     quaternion.w])


    # ---------- 状态 ----------

    def state(self, timeout_s: float = 15.0) -> dict:
        """当前状态，全部从话题来。关节（度）、夹爪开口（米）、法兰与爪尖（基座系）、
        相机位姿（基座系，由标定链算出来，给相机系→基座系那一步用）、末端位姿原值。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._joints is not None and self._end_pose is not None:
                break
        if self._joints is None or self._end_pose is None:
            raise TimeoutError("等不到臂的状态话题（%s / %s）" % (JOINTS_TOPIC, END_POSE_TOPIC))
        # 法兰位姿用**标定链**从关节角算，不用话题上那份 /arm/end_pose。
        #
        # 为什么：/arm/end_pose 是臂自己的 SDK 正解（piper_ctrl_single_node 里
        # `GetArmEndPoseMsgs()`），走的是**名义运动学**，不知道我们标定出来的修正。而
        # IK 解出来的目标是按**带修正**的链算的——两边不在一个坐标系里，视觉算出来的抓取
        # 点会整体偏十几个毫米（实测 17.6 mm，正好把 40 mm 的方块夹空，指头合到 0.1 mm）。
        #
        # 标定台的抓取测试就是这么做的（tools/calibration/eyeinhand/grasp_adapter.py:114，
        # `flange = self.calib.flange(joints)`，那边压根不订阅 /arm/end_pose）——同一台机器
        # 上它能抓住，差别就在这一行。
        #
        # calib.flange 返回的**已经是基座系**，所以不再乘 to_base（那是给话题的世界系用的）。
        flange_base = self.calib.flange(self._joints["joints_deg"])
        tip = self.calib.tip(flange_base)
        return {
            "joints_deg": self._joints["joints_deg"].tolist(),
            "gripper_m": float(self._joints["gripper_m"]),
            "flange": flange_base,
            "tip": tip,
            "camera_pose": flange_base @ self.calib.X,
            "end_pose_world": self._end_pose,
        }

    # ---------- 臂 ----------

    def move_joints(self, joints_deg, gripper_travel=None,
                    tolerance_deg: float = SETTLE_TOL_DEG,
                    timeout_s: float = SETTLE_TIMEOUT_S) -> list:
        """把六轴发到目标角，等它到位，返回**实际**关节角（度）。

        中间**插值走过去**，不把目标一步甩给伺服。仿真里的位置伺服很硬（J5/J6 kp=30，
        实测 0.15 秒就稳住），直接给目标的话整条臂"啪"一下到位——比真机快 5~10 倍，
        看着像闪现。真机那套（calibration/common/robot/direct.py）本来就在做插值
        （`steps = max(100, 100*100/speed)`，每步 sleep 10 毫秒），这里对齐它。

        夹爪的命令值每次都跟着发：话题命令是"只更新提到的关节"，但这里不依赖这个语义，
        统一带全，免得哪次改了解析规则夹爪被复位（旧实现踩过：缺了夹爪项，合爪后被重新张开）。
        """
        if gripper_travel is not None:
            self._gripper_cmd = float(gripper_travel)
        target = np.asarray(joints_deg, dtype=float)
        start = (np.asarray(self._joints["joints_deg"], dtype=float)
                 if self._joints is not None else target)
        # 按**最大的那个关节**要转多少度来定步数，各关节按同一个比例插值——这样它们
        # 同时到，不会有的先到位有的还在走。
        steps = max(1, int(np.ceil(float(np.max(np.abs(target - start))) / MOVE_STEP_DEG)))
        for index in range(1, steps + 1):
            self._publish_command((start + (target - start) * (index / steps)).tolist(),
                                  self._gripper_cmd)
            self._spin_for(MOVE_STEP_S)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._publish_command(target.tolist(), self._gripper_cmd)
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._joints is not None and \
                    float(np.max(np.abs(self._joints["joints_deg"] - target))) <= tolerance_deg:
                break
        if self._joints is None:
            raise TimeoutError("等不到 %s 的关节状态" % JOINTS_TOPIC)
        return self._joints["joints_deg"].tolist()

    def _spin_for(self, seconds: float) -> None:
        """转够这么久再返回。

        不能用一次 `spin_once(timeout_sec=seconds)` 代替：有消息来时它会立刻返回，
        而关节话题是 100 Hz，等于没等——插值的步子会瞬间走完，白插。
        """
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            rclpy.spin_once(self, timeout_sec=min(0.01, remaining))

    def move_tip(self, target_tip, rotation, rounds: int = 3,
                 tolerance_m: float = 0.0015, n_starts: int = 12) -> dict:
        """复用 motion.move_tip 的闭环：解 IK → 走 → 读回实际关节角 → 差多少补一刀。

        闭环里的 calib.flange 只是把话题读回的实际关节角做 FK——臂的模型与运行时逐位
        一致，它不读本地模型的状态；物体、夹爪、相机一概不碰本地模型。
        """
        return motion.move_tip(self.calib, self.move_joints, target_tip, rotation,
                               rounds=rounds, tolerance_m=tolerance_m, n_starts=n_starts)

    def gripper(self, opening_m: float, timeout_s: float = 6.0,
                may_be_blocked: bool = False) -> float:
        """把夹爪开到指定开口宽度（米），等它**动过之后停住**，返回实际开口宽度。

        只看"没在变"会把命令延迟误判成到位（踩过两次：判成 56 mm 夹住，其实一步没合），
        所以先确认反馈离开起始值，再等它连续不变；"连续不变"的窗口要长过伺服振荡的
        衰减时间（见 GRIPPER_STABLE_S 的说明）。

        may_be_blocked: 这次动作**本来就可能被物体顶住**——加压那几步正是如此。这时
            "没动过"是正常结果，不是命令没生效，不必等满超时。默认 False：一般的
            开合仍要确认它真动过。
            踩过：加压时每一步都被方块顶住、`moved` 永远为假，19 步各耗满 6 秒超时，
            合爪那一段从 19 秒涨到 97 秒——抓取按钮要等一分半。
        """
        self._gripper_cmd = max(0.0, float(opening_m)) / 2.0       # 开口宽度 → 单指行程
        deadline = time.monotonic() + timeout_s
        stable_needed = int(GRIPPER_STABLE_S / GRIPPER_SAMPLE_S)
        start = self._joints["gripper_m"] if self._joints is not None else None
        moved, stable, last = False, 0, None
        while time.monotonic() < deadline:
            self._publish_command(self._joints["joints_deg"].tolist()
                                  if self._joints is not None else [0.0] * 6,
                                  self._gripper_cmd)
            rclpy.spin_once(self, timeout_sec=GRIPPER_SAMPLE_S)
            if self._joints is None:
                continue
            now = float(self._joints["gripper_m"])
            if start is None:
                start = now
            if abs(now - start) > GRIPPER_MOVE_M:
                moved = True
            stable = stable + 1 if (last is not None and abs(now - last) < GRIPPER_STABLE_M) else 0
            last = now
            if stable >= stable_needed and (moved or may_be_blocked):
                break
        if self._joints is None:
            raise TimeoutError("等不到 %s 的关节状态" % JOINTS_TOPIC)
        return float(self._joints["gripper_m"])

    def gripper_settle(self, opening_m: float, timeout_s: float = 8.0) -> float:
        """命令夹爪到某个开口，等它**真的停住**才返回——合爪最后一下用它。

        和 gripper() 的差别只在"停住"的判据：那个是"相邻两次采样差 < 0.5 mm"，
        指头以慢速爬行时每一步都小于它，会被当成停住；于是合爪还没闭拢就往下走，
        抬起途中才夹到方块，随即滑脱（现场症状：下爪时夹具没闭拢、边抬边闭）。
        这里要求一个 1.5 秒窗口内的极差 < 0.05 mm —— 慢爬也算动。
        """
        self._gripper_cmd = max(0.0, float(opening_m)) / 2.0       # 开口宽度 → 单指行程
        deadline = time.monotonic() + timeout_s
        window = []                       # [(时刻, 开口)]，只留最近 GRIPPER_SETTLE_WINDOW_S
        while time.monotonic() < deadline:
            self._publish_command(self._joints["joints_deg"].tolist()
                                  if self._joints is not None else [0.0] * 6,
                                  self._gripper_cmd)
            rclpy.spin_once(self, timeout_sec=GRIPPER_SAMPLE_S)
            if self._joints is None:
                continue
            now_t = time.monotonic()
            window.append((now_t, float(self._joints["gripper_m"])))
            while window and now_t - window[0][0] > GRIPPER_SETTLE_WINDOW_S:
                window.pop(0)
            if len(window) >= 10 and now_t - window[0][0] >= GRIPPER_SETTLE_WINDOW_S * 0.9:
                values = [value for _, value in window]
                if max(values) - min(values) < GRIPPER_SETTLE_RANGE_M:
                    break
        if self._joints is None:
            raise TimeoutError("等不到 %s 的关节状态" % JOINTS_TOPIC)
        return float(self._joints["gripper_m"])

    # ---------- 命令 ----------

    def _publish_command(self, joints_deg: list, gripper_travel: float) -> None:
        if self.publisher is None:
            # 没解析到 joint_command 的话，到这里必须炸：静默不动比报错难查得多
            raise RuntimeError(
                "没有可用的 joint_command 话题（atlas 没解析到 primitive/arm/joint_command，"
                "也没配默认值）——臂动不了")
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(ARM_NAMES) + ["gripper"]
        message.position = [float(v) * np.pi / 180.0 for v in joints_deg]
        message.position.append(gripper_travel)
        self.publisher.publish(message)


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9
