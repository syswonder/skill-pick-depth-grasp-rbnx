# Description: 直连 piper SDK 的臂会话——标定/示教这类交互式工具用的低层动作。
#
# 两个部署共用（同一个臂、同一套 CAN 指令），所以这里不读任何部署的配置：
# 口名由调用方给（或自动探测），速度也是。
#
# 为什么不用 s10/robot/arm.py 的 Arm：那是抓取业务的后端（读配置、走 IK、走抓放流程，
# 构造时就会使能并回零），而标定工具要在"人正拖着臂示教"或"刚从示教模式接管"的
# 状态下工作，得直接发 CAN 指令。原先这套动作在几个脚本里各写了一遍
# （使能、接管、关节插值、回零失能），这里收成一份。
import time

import numpy as np
from piper_sdk import C_PiperInterface_V2

from common.robot.canbus import resolve_arm_port

# piper SDK 的定点系数：角度、位置都是以 ×1000 的整数传输
FACTOR = 1000.0
JOINT_COUNT = 6
# 可安全失能的位置：J5 抬 25°，避免零位时整臂压在下限上
HOME_JOINTS_DEG = [0.0, 0.0, 0.0, 0.0, 25.0, 0.0]

# 状态码 → 人话（网页与日志共用）
ARM_STATUS_TEXT = {
    0x00: "正常", 0x01: "急停", 0x02: "无解", 0x03: "奇异点", 0x04: "目标角度超过限",
    0x05: "关节通信异常", 0x06: "关节抱闸未打开", 0x07: "机械臂发生碰撞", 0x08: "拖动示教时超速(拖慢一点)",
    0x09: "关节状态异常", 0x0A: "其它异常", 0x0B: "示教记录", 0x0C: "示教执行", 0x0D: "示教暂停",
    0x0E: "主控NTC过温", 0x0F: "释放电阻NTC过温",
}
# 这两个字段都照 piper_sdk 的枚举抄（ArmMsgFeedbackStatusEnum.TeachingState / CtrlMode）。
#
# 但别拿它们去判断"臂此刻是不是软的、吃不吃得进运动指令"：实测它们跟臂的真实行为对不上。
# 试过用 ctrl_mode == 0x02 挡在回零前面，结果臂明明能正常走，字段却停在 0x02，把回零
# 挡住了，而界面上还看不出被谁挡的。判断依据只认臂的行为，不认字段。
TEACH_STATUS_TEXT = {
    0x00: "未开始示教记录", 0x01: "示教记录中", 0x02: "示教记录已结束", 0x03: "复现轨迹",
    0x04: "复现暂停", 0x05: "复现继续", 0x06: "复现终止", 0x07: "回轨迹起点",
}
CTRL_MODE_TEXT = {
    0x00: "待机", 0x01: "CAN控制", 0x02: "示教模式", 0x03: "以太网", 0x04: "wifi",
    0x05: "遥控器", 0x06: "联动示教", 0x07: "离线轨迹",
}


class DirectArm:
    """一条直连 piper SDK 的臂连接。

    构造只负责连上 CAN 口，不使能、不动臂；每一步动作都要显式调用，
    这样"进示教""接管""回零失能"的顺序在页面流程里一眼可见。
    """

    def __init__(self, port: str | None = None, speed: int = 30):
        """连接 CAN 口并清一次错误码。

        Args:
            port: CAN 口名（如 can4）；`None` 表示自动探测（同臂原语，见 canbus.py）。
            speed: 关节运动速度百分比（1-100），越小越慢越平稳。
        """
        self.speed = speed
        self.port = port or resolve_arm_port()
        self.piper = C_PiperInterface_V2(self.port)
        self.piper.ConnectPort()
        self.piper.JointConfig(clear_err=0xAE)

    # ---------- 读数 ----------

    def status(self, tag: str = ""):
        """读一次臂状态；给了 tag 就顺带打印一行。"""
        st = self.piper.GetArmStatus().arm_status
        if tag:
            print("[%s] ctrl_mode=%s teach_status=%s"
                  % (tag, hex(st.ctrl_mode), hex(st.teach_status)), flush=True)
        return st

    def joints(self) -> list[float]:
        """当前关节角（度）。"""
        j = self.piper.GetArmJointMsgs().joint_state
        return [j.joint_1 / FACTOR, j.joint_2 / FACTOR, j.joint_3 / FACTOR,
                j.joint_4 / FACTOR, j.joint_5 / FACTOR, j.joint_6 / FACTOR]

    def gripper_open_0to1(self) -> float:
        """夹爪开度，0（闭合）~ 1（张开）。"""
        g = self.piper.GetArmGripperMsgs().gripper_state
        return float(np.clip(g.grippers_angle / FACTOR / 100.0, 0, 1))

    def end_pos(self) -> np.ndarray:
        """末端位置 [x, y, z]，单位为米。"""
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        return np.array([ep.X_axis / FACTOR / 1000.0, ep.Y_axis / FACTOR / 1000.0,
                         ep.Z_axis / FACTOR / 1000.0])

    def end_euler_zyx_deg(self) -> list[float]:
        """末端姿态 [RZ, RY, RX]（度）——SDK 回的是 xyz 顺序，这里转成 zyx。"""
        from scipy.spatial.transform import Rotation as R

        ep = self.piper.GetArmEndPoseMsgs().end_pose
        return R.from_euler(
            "xyz", [ep.RX_axis / FACTOR, ep.RY_axis / FACTOR, ep.RZ_axis / FACTOR], degrees=True
        ).as_euler("zyx", degrees=True).tolist()

    # ---------- 使能 / 接管 ----------

    def enable(self, timeout_s: float = 20) -> bool:
        """循环使能六轴，直到全部使能或超时。

        Returns:
            是否在超时内全部使能。
        """
        start = time.time()
        while True:
            self.piper.EnableArm(7)
            msgs = self.piper.GetArmLowSpdInfoMsgs()
            flags = [m.foc_status.driver_enable_status for m in
                     (msgs.motor_1, msgs.motor_2, msgs.motor_3,
                      msgs.motor_4, msgs.motor_5, msgs.motor_6)]
            if all(flags):
                return True
            if time.time() - start > timeout_s:
                return False
            time.sleep(1)

    def set_can_control(self, speed: int | None = None) -> bool:
        """切到 CAN 关节控制模式（实测顺序：先使能，再 MotionCtrl_2）。

        Returns:
            切换后 ctrl_mode 是否为 CAN 控制（0x01）。
        """
        self.piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                                move_spd_rate_ctrl=speed or self.speed, is_mit_mode=0x00)
        time.sleep(0.3)
        st = self.piper.GetArmStatus().arm_status
        if st.ctrl_mode != 0x01:
            print("警告：设置 CAN 控制后 ctrl_mode=%s（期望 0x01）" % hex(st.ctrl_mode), flush=True)
            return False
        return True

    def enable_and_take_control(self, timeout_s: float = 20, speed: int | None = None) -> bool:
        """标定工具的标准起手式：使能 + 切 CAN 控制。"""
        if not self.enable(timeout_s=timeout_s):
            print("使能超时", flush=True)
            return False
        print("arm enabled", flush=True)
        return self.set_can_control(speed=speed)

    def wait_for_teach_exit(self, timeout_s: float = 120) -> bool:
        """人还在拖动示教时等他自己按臂上按钮退出。

        Returns:
            是否已退出示教（超时返回 False）。
        """
        if self.status().teach_status != 0x01:
            return True
        print("检测到臂在示教模式，请按臂上示教按钮退出示教...", flush=True)
        start = time.time()
        while time.time() - start < timeout_s:
            if self.status().teach_status != 0x01:
                print("检测到示教已退出", flush=True)
                return True
            time.sleep(0.3)
        return False

    def take_over(self) -> bool:
        """人退出示教后接管臂：使能 + 切 CAN 控制，并打印结果状态。"""
        if not self.enable():
            print("恢复后使能超时", flush=True)
            return False
        ok = self.set_can_control()
        st = self.status()
        print("接管后 ctrl_mode=%s teach_status=%s"
              % (hex(st.ctrl_mode), hex(st.teach_status)), flush=True)
        return ok

    # ---------- 运动 ----------

    def move_joints(self, target_deg, speed: int | None = None, timeout_s: float = 40,
                    tolerance_deg: float = 1.0) -> None:
        """关节空间插值运动到目标角度（度），到位后返回。

        关节控制模式下 move_spd_rate_ctrl 不一定生效，所以步数按速度反比放大，
        用插值把实际速度压下来。

        Raises:
            RuntimeError: 臂报错、超时，或到位后关节偏差超过 tolerance_deg。
        """
        speed = speed or self.speed
        current_deg = self.joints()
        print("当前关节角: [%s]" % ", ".join("%.2f" % v for v in current_deg), flush=True)
        print("目标关节角: [%s]" % ", ".join("%.2f" % v for v in target_deg), flush=True)

        steps = max(100, int(100 * 100 / speed))
        start = time.time()
        for alpha in np.linspace(0, 1, steps + 1)[1:]:
            q = [c * (1 - alpha) + t * alpha for c, t in zip(current_deg, target_deg)]
            self.piper.JointCtrl(
                joint_1=int(q[0] * FACTOR), joint_2=int(q[1] * FACTOR),
                joint_3=int(q[2] * FACTOR), joint_4=int(q[3] * FACTOR),
                joint_5=int(q[4] * FACTOR), joint_6=int(q[5] * FACTOR),
            )
            time.sleep(0.01)

        while True:
            st = self.piper.GetArmStatus().arm_status
            if st.arm_status != 0x00:
                raise RuntimeError("arm status: %s（%s）"
                                   % (hex(st.arm_status), ARM_STATUS_TEXT.get(st.arm_status, "未知")))
            if st.motion_status == 0x00:
                break
            if time.time() - start > timeout_s:
                raise RuntimeError("运动超时")
            time.sleep(0.02)
        time.sleep(0.5)

        got = self.joints()
        max_err = max(abs(g - t) for g, t in zip(got, target_deg))
        print("到位关节角: [%s]（最大偏差 %.2f°）"
              % (", ".join("%.2f" % v for v in got), max_err), flush=True)
        if max_err > tolerance_deg:
            raise RuntimeError("关节未到位（偏差 %.2f°），可能控制模式异常" % max_err)

    def home_and_disable(self, timeout_s: float = 60) -> bool:
        """回安全位（J5=25°）后失能——工具收尾用；回零失败也会继续失能。

        Returns:
            回零是否成功（失败也会失能，只是臂停在半路）。
        """
        try:
            self.move_joints(HOME_JOINTS_DEG, speed=20, timeout_s=timeout_s)
            ok = True
        except Exception as e:
            print("回零失败:", e, flush=True)
            ok = False
        self.piper.DisableArm()
        self.piper.GripperCtrl(gripper_code=0x02)
        print("已回零并失能" if ok else "已失能（但没回到安全位）", flush=True)
        return ok

    # ---------- 夹爪 ----------

    def open_gripper(self, effort: int = 5000) -> None:
        """张开夹爪（1.0 = 全开）。"""
        self.piper.GripperCtrl(gripper_angle=int(1.0 * 100 * FACTOR), gripper_effort=effort,
                               gripper_code=0x03, set_zero=0)

    def close_gripper(self, effort: int = 2000) -> None:
        """合上夹爪（0.0 = 全合）。

        夹爪尖标定要靠它：两根手指并起来才有一个确定的尖，标定过程中不能再张开。
        力矩给得比抓东西小，合到底之后堵转的力小一点。
        """
        self.piper.GripperCtrl(gripper_angle=0, gripper_effort=effort,
                               gripper_code=0x03, set_zero=0)

    def set_gripper(self, ratio: float, effort: int = 3000) -> None:
        """把夹爪开到 ratio（0 = 全合，1 = 全开）。

        开合两档之间的那些位置只能这么给：试夹不同大小的东西时用得上。
        """
        self.piper.GripperCtrl(gripper_angle=int(np.clip(ratio, 0.0, 1.0) * 100 * FACTOR),
                               gripper_effort=effort, gripper_code=0x03, set_zero=0)

    # ---------- 拖动示教 ----------

    def enter_teach_mode(self) -> None:
        """进入拖动示教（臂松弛，可手拖）。

        先把示教状态清干净：上一次会话被杀掉时控制板可能残留"记录中"状态，
        此时直接发 0x01 会被当成切换关闭。
        """
        self.piper.MotionCtrl_1(0x00, 0x00, 0x02)   # 结束示教记录
        time.sleep(0.3)
        self.piper.MotionCtrl_1(0x02, 0x00, 0x00)   # 恢复（回位置速度控制）
        time.sleep(0.3)
        self.piper.MotionCtrl_1(0x00, 0x00, 0x01)   # 开始拖动示教
        st = self.status()
        for _ in range(10):
            if st.teach_status == 0x01:
                break
            time.sleep(0.3)
            st = self.status()
        print("teach cmd sent: ctrl_mode=%s teach_status=%s"
              % (CTRL_MODE_TEXT.get(st.ctrl_mode, hex(st.ctrl_mode)),
                 TEACH_STATUS_TEXT.get(st.teach_status, hex(st.teach_status))), flush=True)
        if st.teach_status != 0x01:
            print("警告：反馈未确认进入示教模式——若拖动阻力大，请按住臂上的示教按钮再拖", flush=True)

    def exit_teach_mode(self) -> None:
        """退出拖动示教，臂停在当前位置保持。"""
        self.piper.MotionCtrl_1(0x00, 0x00, 0x02)   # 结束拖动示教
        time.sleep(0.3)
        self.piper.MotionCtrl_1(0x02, 0x00, 0x00)   # 恢复（回位置速度控制）
        print("已退出示教模式，臂在当前位置保持", flush=True)

    # ---------- 收尾 ----------

    def close(self) -> None:
        """断开 CAN 连接（不动臂、不失能）。"""
        self.piper.DisconnectPort()
