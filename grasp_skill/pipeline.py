# Description: 一条抓取流程：悬停 → 拍照 → 检测 → 平面 → 计划 → 下爪 → 合爪 → 抬起。
# **真机这一份**：走位回调（servo）由外面注入，抓取几何（grasp / motion）原地不动。
# 状态一律从 client（话题）来，本地模型只出现在 IK 求解里。
#
# **"看"那一块不在本文件里，也不再在技能里**：检测、承托面拟合、顶面提取、"框和角度
# 换算成三维"都在视觉服务 `robonix/service/perception/grasp_geometry/locate`（见
# grasp_skill/vision_service.py）。本文件只做两件事：把"找什么"发过去、把回来的
# **相机系**结果搬到基座系。视觉算法改了重启服务即可（一秒），不用重启 rbnx。
#
# **本文件不订阅任何图像话题**：帧在服务那边取，技能端没有相机依赖。
import json
import logging
import time
from pathlib import Path

import numpy as np

import grasp
import motion
from grasp_skill import vision_service

# 多观察位扫描的规划器（纯 numpy，不碰硬件/相机/服务）：标称拍照位"这一眼没量出来"时，
# 它算几个换着看的观察位。**模块缺失不许拖垮抓取**——扫描是补救措施，不是必要条件：
# 旧包里没有这个文件时整条流程照旧，只是少一层"换个角度再看"。
try:
    import observe_scan
except ImportError:                              # 只可能是没同步到旧包的部署
    observe_scan = None

# 与技能其它模块同一个 logger，落在同一个技能日志里（见 atlas_bridge）
log = logging.getLogger("depth_grasp")


def to_base(camera_base, point) -> np.ndarray:
    """相机系的一点 → 基座系。方向（法线/长轴/闭合）不走路，要调用方自己转。"""
    return (np.asarray(camera_base, dtype=float) @ np.r_[np.asarray(point, dtype=float), 1.0])[:3]


def orientation_gap_deg(a, b) -> float:
    """两个姿态之间差多少度（把旋转差折算成一个转轴一个角）。"""
    cosine = (np.trace(np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def retryable_measurement_failure(error) -> bool:
    """这一眼失败，值不值得**换个观察位再看一眼**。

    只有"服务收到了帧、但这一眼没量出来"（物体不在画面/承托面量不到）值得换角度；
    连不上或超时（transport）、服务非 200（server）、传参不合法（request）换多少角度
    都一样，白走几趟几秒还耽误时间。分类在 vision_service.VisionError.kind 里。

    kind 是视觉客户端给的；没有 kind 时退回文案判断（失败分支里那句
    "这一眼没量出来" 是测量类失败的前缀），保证不同版本的客户端都能分类。
    """
    kind = getattr(error, "kind", None)
    if kind is not None:
        return kind == "measurement"
    return "这一眼没量出来" in str(error)


# 锚点/扫描位去重阈值（度）：与已经看过的位姿每个关节都差不到这么多，就算同一个位姿，
# 不重复走过去——多锚点时人可能把锚点写成标称位、或某个扫描位刚好撞上，重复走纯浪费。
_ANCHOR_SAME_DEG = 1.0

OPENING_M = 0.056              # 张开：开口宽度（米）——没有夹爪几何时的兜底值
                              # （真机上用 gripper.opening_mm 那句满开口，见 _opening_m）
# 合爪三段走：远段大步、接近接触后小步逼近、顶住之后照捏力加压。
# 捏力 = kp ×（实际开口 − 命令开口）（两指合计；顶住时指头不动，位置伺服的力就是
# 这个，kp 见 piper.xml 的夹爪执行器）。实测 0.34 N 抬不起方块、1.68 N 抬得起。
CLOSE_COARSE_OPENINGS_M = (0.044, 0.0425)      # 远段大步（接触宽度 ~40 mm 之上）
CLOSE_FINE_STEP_M = 0.0005                     # 接近接触后的小步，也是加压的步长
CLOSE_FINE_LIMIT_M = 0.034                     # 小步的终点（低于任何可能的接触宽度）
CLOSE_BLOCKED_TOL_M = 0.00025                  # 开口没跟着命令走这么多 = 被顶住
CLOSE_SQUEEZE_N = 2.5                          # 顶住之后加压到的捏力（牛顿，两指合计）
                                               # 1.2 → 2.5：原来的 1.2 压在**自己量的阈值
                                               # 之下**（注释里那句"0.34 N 抬不起、1.68 N
                                               # 抬得起"就是它自己的标定），实测一轮 `pinch_n`
                                               # 正好 1.197 N：抬起 127 mm 成功，回位途中滑掉。
                                               # 2.5 = 1.68 × 1.5，留一半余量。加压量按
                                               # 干涉量算 = 2.5/200 = 12.5 mm，远在夹爪
                                               # forcerange（±10 N）之内。
                                               # 上限受的不是力，是接触几何：指头没落到
                                               # 平行面上（骑在棱上），再大的力也只是把物体
                                               # 更快挤出去。
SQUEEZE_ROUNDS = 3                             # 加压最多重来几轮，每轮按**实测开口**重压
                                               # （见合爪那段的说明：接触可能先停在离物体
                                               # 几毫米的地方，一次压不到位）
GRIPPER_KP = 200.0                             # 夹爪位置伺服的刚度，与 piper.xml 一致
# 注：原来还有 GRABBED_MIN_FRACTION（"握着时开口该多宽"）和 BLOCK_AT_TIP_MM（方块离爪尖
# 多近算还在手里）——那两条都要**物体位姿源**，真机没有，判据改用 client.holding()。
DESCEND_MAX_JOINT_MOVE_DEG = 70.0
# 下爪那一步"算到位"的判据（米）。比全局的 SETTLED_M（2 mm）松，理由是实测踩过的：
# 从**实到的**悬停位（偏 5 mm/5°）解下爪，IK 只能收敛到差 2.3 mm；而判据是 2.0 mm，
# 于是**整个解被判废、一步不走**——现场就是"悬停到位了、下爪一动不动"。
# 差这 2.3 mm 换来的是什么：物体宽 74 mm、开口 102 mm、指板 72 mm，照样夹得住；
# 而且 move_tip 有两轮闭环、合爪那一步还有接触检测兜底。0.3 mm 换"完全不干"，不划算。
# 判据**不拍常数**，按爪和物体自己的尺寸现算：见 grasp.descend_settled_m()。
# 实测踩过：拍 2 mm → 差 2.3 mm 一步不走；拍 5 mm → 下一轮差 6.84 mm 还是不走。
# 差多少算"还能夹住"是物理决定的——(开口 − 物体宽)/2，跟这一轮量到的宽度走。
# 下爪那一步允许的关节移动上限。悬停与抓取两个姿态本就不同族（实测这一落要 41~48°）；
# 真换到腕部另一支上去要做 104° 起步的翻转（现场：J6 拧了 280°、指头撞上方块、爪尖
# 差 21.5 mm，合爪时开口压到 29.6 mm 把方块挤开）。两档之间空得很，70 正好落在中间。
# 悬停那一步**只**保一条判据：从这儿能一步下到抓取点（descends_from）。下面这三条是
# 位置/姿态精度，在悬停这儿冗余——下爪是按臂实到的关节角重新解 IK 的，悬停偏一点会被
# 它吃掉。放得比抓取那一步宽，是为了别让"路过"的姿势把整轮卡死。
HOVER_LATERAL_TOL_M = 0.015
HOVER_ALONG_TOL_M = 0.030
HOVER_CLOSING_TOL_DEG = 20.0
# 悬停这一步工具轴允许歪多少。**和抓取那一步的 `grasp.MAX_OBLIQUITY_DEG`（12°）分开**：
# 那个管的是"两指别斜着切进去"（抓握质量，会变成线接触、夹不稳），悬停只是路过，
# 歪一点不影响抓，而歪一点往往正好是"这个位置竖直够不着"的解药。
HOVER_OBLIQUITY_DEG = 25.0
HOVER_TOLERANCE_M = 0.005      # 悬停到位判据。比下爪那道（1.5 mm）宽：悬停只是路过，
                               # 下爪的目标是按**实到**的爪尖算的，差几毫米不影响抓；按
                               # 1.5 mm 要求它，实测要多走一整轮（4 s 移动 + 1 s 解算）。
# 注：环带、有效深度占比那些**视觉参数**（RING_* / MIN_FILLED_FRACTION）不在这儿了——
# 跟着算法进了视觉服务（service/perception/grasp_geometry）。本文件不该再有视觉的调参。
DESCEND_ABORT_M = 0.015
# 合爪时"顶住"的开口比物体宽出这么多，就不算夹住：那只说明指头没贴到物体上
# （骑在方块边上、或先撞上了别的），挤压是白压、抬起是抬空手。实测：正常夹住在
# 42.6 mm（方块 40 mm，指垫 2.6 mm）；失败那轮停在 52.2 mm（宽 12.2 mm）——
# 现场看着就是"夹具没闭合"，而程序以为夹住了。
CLOSE_CONTACT_MAX_GAP_MM = 5.0
# 抬起（走回悬停位）按关节角判到位：伺服自己的到位判据是 1.0°，留点余量给 1.5°。
LIFT_MAX_ANGLE_ERROR_DEG = 1.5
# 下爪差这么多（米）就判定"根本没到位"，**不再合爪**，整轮作废并如实报出来。
# 现场：爪尖比计划高了 21 mm（指头被方块顶住下不去），伺服每个动作顶满 30 秒超时、
# 三轮共 89 秒，然后照样合爪把开口压到 34.1 mm（方块 40 mm）——从方块边上挤过去，
# 把方块推走 26 mm，什么也没夹到。够不到就别夹，比"顶着挤"干净得多。
#
# 10 → 15 mm：原来那道闸正好压在伺服的误差地板上。关节 frictionloss=0.3、位置伺服
# 没有积分项（力 = kp×偏差），稳态偏差就是 0.3/kp —— J1~J3 是 0.215°、J4 0.43°、
# J5/J6 0.573°。这个场景的方块离基座 0.574 m（斜面在 world x=0.24、基座在 −0.335），
# **1° 正好就是 10.0 mm**：闸门与地板是同一条线，于是同一副姿态轮轮在 9~12 mm 之间
# 跳。实测连续几轮 9.0 / 10.2 / 10.2 / 10.3 / 10.8 / 11.6 mm——9.0 那轮夹起来了，
# 10.2 那轮报"被顶住"。而这个偏差是**静摩擦堵住**，不是没等够，等多久都还是这么多
# （爬升靠 move_tip 那第二刀，但它解出来的关节移动量 0.58° 和摩擦带同量级，顶不开就
# 又咬住，收不下去）。15 mm 落在"伺服地板（≤12）"与"真被顶住（21.5）"之间。
# 注意它只是把闸门挪到地板之上，地板本身没动：要真降下来得减小力臂（方块挪近基座），
# 或者降 frictionloss、提 J1~J3 的 kp。


class Pipeline:
    """一条抓取流程。走位回调（servo）由外面注入；"看"那一步打给视觉服务。"""

    def __init__(self, client, calib, servo, scene: dict, out_dir: Path,
                 vision=None, observe_scan_config: dict | None = None):
        self.client = client
        self.calib = calib
        # 视觉客户端（grasp_skill.vision_service.VisionClient）。"看"那一步（取帧、
        # 检测、平面、顶面、框→三维）全在视觉服务里，这里只负责调它、搬坐标系。
        self.vision = vision
        # 多观察位扫描的覆盖项（部署清单的 observe_scan 块）。**这里不合并默认值**——
        # 合并发生在规划那一步（observe_scan.merged_config），好处是清单里只写要改的那几个键，
        # 而且"哪些键有默认值"只有插件那一处定义。
        self.observe_scan_config = dict(observe_scan_config or {})
        # 阶段计时：_mark_stage 是流水线里确定的阶段边界（1-hover / 2-detect /
        # 3-descend / 4-closed / 5-lift），在那里打点，相邻两点的时间差就是这个
        # 阶段花了多久。想知道"一轮的时间去哪了"看这个，别看函数级的 profiler——
        # 那个只知道"16 秒花在等 ROS 回调上"，不知道是哪个阶段在等。
        self._t0: float | None = None
        self._timings: list = []
        self._capture = {}
        self._run_started = None
        self.servo = servo                    # move(joints_deg) -> 实际关节角（度）
        self.scene = scene                    # real_adapter.scene_from_config 的产物
        # out_dir 给 None 就整轮不落盘（技能默认不落；开发台上照旧落）
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.to_base = np.linalg.inv(getattr(calib, "base_pose", np.eye(4)))
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def observe(self) -> dict:
        """回拍照位看一眼、只量不动：走到固定姿势、张开夹爪、量一眼并给出计划结论。

        臂只走"回观察位"这一步（外加张开夹爪），**不做任何抓取动作**。量的是这一眼的
        检测框、顶面中心、承托面的朝向、闭合方向、投影宽度，以及照这些量算出来的计划
        过不过关——给抓之前确认现场、或者抓失败后查原因用。
        """
        self._t0 = time.monotonic()
        achieved = self.servo(self._photo_pose())
        self.client.gripper(self._opening_m())
        self._mark_stage("photo")
        joints = [round(float(v), 2) for v in achieved]
        result = {"ok": False, "action": "observe",
                  "environment": self.scene.get("environment"),
                  "problems": [],
                  "stages": {"photo": {"joints_deg": joints,
                                       "look_joints_deg": joints}}}
        look = self._look("photo")
        result["stages"]["look_photo"] = look.get("report")
        if not look["ok"]:
            result["problems"].append("在拍照位没看准：" + look["why"])
            return result
        result["measurement"] = {
            "top_point_m": np.round(np.asarray(look["top_point"], dtype=float), 4).tolist(),
            "support": {
                "point_m": np.round(np.asarray(look["plane"]["point"], dtype=float), 4).tolist(),
                "normal": np.round(np.asarray(look["plane"]["normal"], dtype=float), 4).tolist(),
            },
            "long_axis": np.round(np.asarray(look["long_axis"], dtype=float), 4).tolist(),
            "closing": np.round(np.asarray(look["closing"], dtype=float), 4).tolist(),
            "footprint_width_m": look.get("footprint_width"),
            "request_id": look.get("request_id"),
        }
        if getattr(self.calib, "gripper", None) is None:
            result["problems"].append("计划没通过：这份标定里没有夹爪尺寸（gripper_size）")
            return result
        plan = grasp.plan(look["top_point"], look["plane"], look["long_axis"],
                          look["closing"], look["footprint_width"], self.calib.gripper)
        result["stages"]["plan"] = {
            "object_height_mm": plan["object_height_m"] * 1000,
            "tip_height_mm": (None if plan["tip_height_m"] is None
                              else plan["tip_height_m"] * 1000),
            "hover_tip_m": np.round(plan["hover_tip"], 4).tolist(),
            "grasp_tip_m": (None if plan["grasp_tip"] is None
                            else np.round(plan["grasp_tip"], 4).tolist()),
            "footprint_width_mm": (None if plan.get("footprint_width_m") is None
                                   else plan["footprint_width_m"] * 1000),
            "problems": list(plan["problems"]),
        }
        if not plan["ok"]:
            result["problems"].append("计划没通过：" + "；".join(plan["problems"]))
            return result
        result["ok"] = True
        return result

    def _target_name(self) -> str:
        """目标名字，给视觉服务当检测的提示词。

        scene 里那份是 `object_<名字>`（real_adapter 拼的），
        这里去掉前缀——提示词里要的是人能看懂的名字。
        """
        return str(self.scene["target"]["name"]).removeprefix("object_")

    def _opening_m(self) -> float:
        """张开到多大：有夹爪几何就用**满开口**，没有才退回常量。

        真机上 56 mm 这个兜底值对稍大一点的物体就是"没张开"——指头会蹭着物体下去
        （实测：下爪时夹爪没张到最大）。满开口是标定台「夹爪」页量的那个数。
        """
        full = (self.calib.gripper or {}).get("opening_mm") if getattr(self.calib, "gripper", None) else None
        return max(OPENING_M, float(full) / 1000.0) if full else OPENING_M

    def opening_m(self) -> float:
        """张开到多大（米）：有夹爪几何用满开口，没有就退常量。

        `put_down` 也要用它（"开到最大"必须和抓取那套同一个口径），所以给它一个
        公开入口，不在编排层重复算。
        """
        return self._opening_m()

    def _photo_pose(self) -> list:
        """拍照位：**固定姿势**——不解 IK、不看物体在哪、不留任何东西给后面的步骤。

        真机那份是标定时教好的（artifacts/poses/observe_pose.json，与部署清单的
        roboarm_ik.init_joints_deg 同源）。仿真那边有它自己的一份（写在 tools/grasp_tilt
        的 sim_ik 里）：两边相机装法一致，但场景坐标不同，所以各教各的。
        """
        joints = getattr(self.calib, "observe_joints", None)
        if joints is None:
            raise RuntimeError("这份标定里没有拍照位（observe_joints）")
        return [float(v) for v in joints]

    def _look(self, label: str) -> dict:
        """让视觉服务量一次：回这一眼看到的（框、顶面、承托面、朝向、投影宽度）。

        量的活全在服务里（`service/perception/grasp_geometry`）：它自己订阅相机、取
        最新一帧、跑检测与几何，把**相机系**结果交回来。本方法只把结果搬到基座系——
        **本文件里一行像素都不碰**。

        每一次"看"都是一次独立测量，不读上一步留下的任何量。失败不抛异常，
        回 {"ok": False, "why": ...}，由调用方写进 problems。
        """
        # 量之前确认夹爪是张开的：手指挡在物体前面，这一眼就白量了。已经张着就不动它
        # ——再发一次命令要等"动过之后停住"，指头本来就在位，得白等一次超时。
        if self.client.state()["gripper_m"] < self._opening_m() - 0.002:
            self.client.gripper(self._opening_m())
        state = self.client.state()
        camera_base = state["camera_pose"]
        # 留下这一次的现场：`state` 里含相机位姿（相机系→基座系要用的那一乘）。
        # 原始帧不在这儿——帧在视觉服务那边，技能端不订阅相机。
        self._capture = {"state": json.loads(json.dumps(state, default=_json_default))}
        if self.vision is None:
            why = "视觉客户端没接上（技能没有走 on_activate 建 VisionClient）"
            self._capture["why"] = why
            return {"ok": False, "why": why, "kind": "server", "retryable": False}
        try:
            measured = self.vision.locate(target=self._target_name())
        except vision_service.VisionError as error:
            # 服务连不上、或者这一眼量不出来：**如实报，不退回默认值**。拿旧值或默认值
            # 往下走，就是"看起来像算法坏了"的来源（这一条踩过）。
            # retryable 告诉调用方"换个观察位有没有意义"：只有"这一眼没量出来"值得换；
            # 连不上/服务报错/传参错误换多少角度都一样（见 retryable_measurement_failure）。
            self._capture["why"] = str(error)
            return {"ok": False, "why": str(error), "kind": getattr(error, "kind", None),
                    "retryable": retryable_measurement_failure(error)}

        # 相机系 → 基座系。点要转也要平；方向（法线、长轴、闭合）只转；宽度是标量不用动。
        rotation = np.asarray(camera_base, dtype=float)[:3, :3]
        top_point = to_base(camera_base, measured["top_point"])
        plane_point = to_base(camera_base, measured["support"]["point"])
        normal = rotation @ np.asarray(measured["support"]["normal"], dtype=float)
        normal = normal / np.linalg.norm(normal)
        long_axis = rotation @ np.asarray(measured["long_axis"], dtype=float)
        long_axis = long_axis / np.linalg.norm(long_axis)
        closing = rotation @ np.asarray(measured["closing"], dtype=float)
        closing = closing / np.linalg.norm(closing)

        bbox = [float(v) for v in measured["bbox"]]
        diagnostics = measured.get("diagnostics") or {}
        self._capture["detector"] = diagnostics
        # 这一眼服务给了承托面：记下高度，观察位扫描拿它当先验（见 _estimate_surface_z_from_capture）。
        self._capture["surface_z_m"] = float(plane_point[2])
        self._mark_stage(label)
        return {
            "ok": True,
            "obb": {"bbox": bbox,
                    "center_px": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)},
            "top_point": np.asarray(top_point, dtype=float),
            "plane": {"point": np.asarray(plane_point, dtype=float), "normal": normal},
            "long_axis": np.asarray(long_axis, dtype=float),
            "closing": np.asarray(closing, dtype=float),
            "footprint_width": measured.get("footprint_width"),
            "camera_base": camera_base,
            "report": diagnostics,
            "request_id": measured.get("request_id"),
        }

    @staticmethod
    def _pose_seen(joints_deg, seen_poses) -> bool:
        """这个位姿是不是已经走过/看过了：每个关节都差不到 `_ANCHOR_SAME_DEG` 度就算。

        多锚点时用得上：锚点可能就写成标称位、某个锚点周围的扫描位也可能跟前面撞上。
        重复走一遍既费时间又没有任何新信息。
        """
        return any(len(tried) == len(joints_deg)
                   and max(abs(float(a) - float(b)) for a, b in zip(joints_deg, tried))
                   <= _ANCHOR_SAME_DEG
                   for tried in seen_poses)

    def _estimate_surface_z_from_capture(self, config) -> float | None:
        """估一个"支撑面高度"（基座系 z，米），给多观察位扫描当先验；估不出返回 None。

        来源优先级：
          1. 视觉结果里服务给的承托面（`_look` 成功时记下的 `surface_z_m`）；
          2. 兼容离线/测试注入的原始深度帧（真实运行里本技能**不订阅相机**，走不到）；
          3. None —— 调用方退回 scene 里那个常量承托面（按 table_height_m 拼的）。

        真实运行里第 1 条帮不上忙：扫描只在"这一眼没量出来"时触发，而那一眼服务不返回
        承托面（失败响应只有 message + result_json={ok:false, why}）。所以真机上这一层
        目前实际会退到配置常量——见 CAPABILITY.md / README 的已知限制与 TODO。
        """
        surface_z = self._capture.get("surface_z_m")
        if surface_z is not None:
            try:
                return float(surface_z)
            except (TypeError, ValueError):
                pass
        depth = self._capture.get("depth")
        capture_state = self._capture.get("state") or {}
        camera_pose = capture_state.get("camera_pose")
        camera_matrix = getattr(self.calib, "camera_matrix", None)
        if depth is None or camera_pose is None or camera_matrix is None:
            return None
        if observe_scan is None:
            return None
        try:
            value = observe_scan.estimate_surface_z(
                depth, np.asarray(camera_pose, dtype=float),
                np.asarray(camera_matrix, dtype=float), config)
        except Exception as error:                # noqa: BLE001 —— 估不出由调用方退回去
            log.warning("估支撑面高度出错（%s: %s）", type(error).__name__, error)
            return None
        return None if value is None else float(value)

    def _observe(self, result: dict) -> tuple:
        """第 1、2 步：去拍照位看一眼；看不准、且是"这一眼没量出来"那类失败，才换几个
        自动规划的观察位再找，找到即停；全看不准则回标称位。

        观察位分两层（**多锚点**）：
          - 锚点 = 一个"合格观察姿势"。第一个锚点是标称拍照位（标定教好的），之后的锚点
            来自清单 `observe_scan.anchors_deg`（人示教/指定的其它观察位）；
          - 每个锚点周围再规划最多 `max_views` 个扫描位。
        顺序：标称位扫完 → 锚点 1 看一眼、再围它扫 → 锚点 2 …。每个锚点自己先看一眼，
        它看准了就不扫它周围（人给的姿势本来就是好视角）。**每个锚点的支撑面高度都拿
        它自己那一帧深度重估**：标称位看到地面、第二个观察位看到桌面时，两者差 0.7 m，
        用同一个高度去规划会把锚点周围的扫描位全打到地面上去。

        为什么**先**标称位：那一站是标定教好的，相机离物体约 0.6 m，深度和框都量得最准；
        换观察位是**补救**，不是常规路径——每换一次都要动臂，代价好几秒。
        为什么只在"测量失败"时才换：连不上服务 / 服务回了非 200，换个角度不会变好，
        白走几趟反而耽误时间（见 retryable_measurement_failure）。

        Returns:
            (look, scan_stage)。look 是最后那一眼的结果（成功时就是成功那一份）；
            scan_stage 为 None 表示"没扫"（标称位就看准了），否则是一份记录——扫没扫、
            支撑面按多高算、每个锚点走到没有/规划了哪些位姿/每个位姿成没成、赢的是哪个
            锚点的第几个。没扫的时候也把键摆齐（attempts/anchors 为空），读日志的人不用猜。
        """
        # ---- 1. 去拍照位：固定姿势；**张开夹爪**（手指挡在镜头前就没法拍了） ----
        photo_joints = self._photo_pose()
        self.servo(photo_joints)
        open_actual = self.client.gripper(self._opening_m())
        result["stages"]["photo"] = {
            "target_joints_deg": [round(float(v), 2) for v in photo_joints],
            "actual_joints_deg": [round(float(v), 2)
                                  for v in self.client.state()["joints_deg"]],
            "opening_mm": round(open_actual * 1000, 1)}

        started = time.monotonic()

        def stage(**overrides) -> dict:
            """扫描记录的统一形状：没扫时也把键摆齐，读的人不用猜哪个键可能不在。"""
            record = {"enabled": True, "triggered": False, "surface_z_m": None,
                      "anchors": [], "winning_anchor": None,
                      "planned": [], "notes": [], "attempts": [],
                      "winning_index": None, "returned_to_nominal": False,
                      "elapsed_s": None}
            record.update(overrides)
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            return record

        def snap_look_joints() -> None:
            """记下**这一眼是在哪副关节角上看的**（照片位，还是某个备用观察位）。"""
            result["stages"]["photo"]["look_joints_deg"] = [
                round(float(v), 2) for v in self.client.state()["joints_deg"]]

        # ---- 2. 拍照检测：物体在哪、承托面朝哪 ----
        look = self._look("2-look-photo")
        result["stages"]["look_photo"] = look.get("report")
        snap_look_joints()
        if look["ok"]:
            return look, None                     # 标称位就看清了：多一步都不走

        if observe_scan is None:
            return look, stage(
                enabled=False,
                notes=["这个包里没有 observe_scan 模块（多观察位扫描）——"
                       "只按标称位这一眼处理，不换观察位"])
        try:
            config = observe_scan.merged_config(self.observe_scan_config)
        except Exception as error:                # noqa: BLE001 —— 配置坏了不该拖垮整轮
            log.warning("observe_scan 配置不合法（%s: %s），本次不扫",
                        type(error).__name__, error)
            return look, stage(notes=["observe_scan 配置不合法（%s: %s），本次不扫"
                                      % (type(error).__name__, error)])
        if not bool(config.get("enabled", True)):
            return look, stage(enabled=False,
                               notes=["部署配置里关了多观察位扫描（observe_scan.enabled=false）"])
        if not look.get("retryable"):
            return look, stage(notes=[
                "这一眼失败不是\"没量出来\"那一类（kind=%s），换观察位也没用，不扫"
                % (look.get("kind") or "未提供（旧客户端）")])

        # 支撑面高度先验：从**刚拍的那一帧**深度里估。不读配置里的 table_height_m——
        # 实测它比现场地面高 0.39 m（配置是旧的），拿它当平面筛出来的位姿全在错误高度上。
        # 每个锚点还会在走到位后用它自己那一帧重估一次（见 _estimate_surface_z_from_capture）。
        surface_z = self._estimate_surface_z_from_capture(config)
        notes = []
        if surface_z is None:
            # 兜底：scene 里那个常量承托面（real_adapter 按 table_height_m 拼的，只进诊断）。
            # 它可能过时，所以只在"深度估不出来"时用，并在 notes 里写明来源。
            ramp = self.scene.get("ramp") if isinstance(self.scene, dict) else None
            plane = ramp.get("plane") if isinstance(ramp, dict) else None
            point = plane.get("point") if isinstance(plane, dict) else None
            if isinstance(point, (list, tuple)) and len(point) >= 3:
                surface_z = float(point[2])
                notes.append("这一帧深度估不出支撑面高度，退回配置里的常量平面 z=%.3f m"
                             % surface_z)
        if surface_z is None:
            notes.append("这一帧深度里估不出支撑面高度，配置里也没有可用的承托面常量，"
                         "没法筛观察位，不扫")
            return look, stage(notes=notes)

        # ---- 多锚点：标称位（已看过）→ 绕标称位扫 → 锚点 1 看一眼 → 绕锚点 1 扫 → 锚点 2 …
        #      （锚点来自清单 observe_scan.anchors_deg；不配就是原来"只在标称位周围扫"）
        anchor_list = [("标称位", list(self._photo_pose()))]
        seen_poses = [list(self._photo_pose())]
        for order, joints in enumerate(config.get("anchors_deg") or [], start=1):
            try:
                joints = [float(v) for v in joints]
            except (TypeError, ValueError):
                notes.append("anchors_deg 第 %d 组不是 6 个数，跳过" % order)
                continue
            # 防御性校验：真 merged_config 会先把这里校一遍（写错整份配置报错），但测试桩/
            # 旧模块给的 config 未必。走到一个 6 个角以外的位姿会直接炸伺服，不如不扫。
            if len(joints) != 6 or not all(np.isfinite(joints)):
                notes.append("anchors_deg 第 %d 组不是 6 个有限的关节角，跳过" % order)
                continue
            if self._pose_seen(joints, seen_poses):
                notes.append("anchors_deg 第 %d 组跟前面的锚点几乎是同一个位姿，跳过" % order)
                continue
            anchor_list.append(("锚点 %d" % order, joints))

        anchors = []
        attempts = []
        winning_anchor = None
        winning_view = None
        moved = False
        stop_reason = ""
        for anchor_index, (anchor_name, anchor_joints) in enumerate(anchor_list):
            record = {"anchor_index": anchor_index, "name": anchor_name,
                      "joints_deg": [round(float(v), 2) for v in anchor_joints],
                      "look": None, "surface_z_m": None, "planned": [], "notes": [],
                      "attempts": [], "winning_view": None}
            anchors.append(record)
            if anchor_index > 0:
                # 先走到这个锚点看一眼：它是人给的"合格观察位"，它自己看准了就不用再扫。
                if self._pose_seen(anchor_joints, seen_poses):
                    # 这个位姿前面已经看过（比如锚点就是标称位周围某个扫描位）：
                    # 再看第二眼没有新信息，直接围它扫。
                    record["look"] = {"skipped": "这个位姿前面已经看过，不再看第二眼"}
                    record["notes"].append("锚点跟已经看过的位姿相同：跳过它自己那一照，只围它扫")
                else:
                    self.servo(anchor_joints)
                    moved = True
                    seen_poses.append(list(anchor_joints))
                    look = self._look("2-look-anchor-%d" % anchor_index)
                    result["stages"]["look_photo"] = look.get("report")
                    snap_look_joints()
                    record["look"] = {"ok": bool(look["ok"]), "why": look.get("why")}
                    if look["ok"]:
                        winning_anchor, winning_view = anchor_index, -1
                        log.info("%s 自己就看准了，不用再扫", anchor_name)
                        break
                    if not look.get("retryable"):
                        record["look"]["stopped"] = "失败不可重试，停止扫描"
                        stop_reason = ("%s 看一眼失败且不可重试（kind=%s），停止扫描"
                                       % (anchor_name, look.get("kind")))
                        log.info("%s 看一眼失败且不可重试（kind=%s），停止扫描",
                                 anchor_name, look.get("kind"))
                        break
            # 每个锚点用它自己那一帧的支撑面：走到锚点后拍的那一眼，看到的是地面还是桌面，
            # 决定了围它扫的位姿该往多高打。跳过那一照（重复位姿）时没有新帧，沿用上一个。
            anchor_surface_z = surface_z
            if anchor_index > 0 and record["look"] and record["look"].get("ok") is False:
                estimated = self._estimate_surface_z_from_capture(config)
                if estimated is None:
                    record["notes"].append("这个锚点那一帧估不出支撑面高度，沿用上一个面高 "
                                           "z=%.3f m" % surface_z)
                else:
                    anchor_surface_z = estimated
                    record["notes"].append("按这个锚点自己那一帧重估支撑面高度 z=%.3f m"
                                           % estimated)
            record["surface_z_m"] = (None if anchor_surface_z is None
                                     else round(float(anchor_surface_z), 4))
            try:
                plan = observe_scan.plan_views(self.calib, anchor_joints,
                                               float(anchor_surface_z), config)
            except Exception as error:            # noqa: BLE001 —— 规划器炸了也不能拖垮抓取
                log.warning("规划 %s 的观察位出错（%s: %s），跳过这个锚点",
                            anchor_name, type(error).__name__, error)
                record["notes"].append("规划观察位出错（%s: %s），跳过这个锚点"
                                       % (type(error).__name__, error))
                continue
            record["planned"] = list(plan.get("views") or [])
            record["notes"].extend(plan.get("notes") or [])
            if not plan.get("ok") or not record["planned"]:
                if plan.get("ok"):
                    record["notes"].append("规划器没给出可用的备用观察位")
                continue
            for view_index, view in enumerate(record["planned"]):
                target = [float(v) for v in view.get("joints_deg") or []]
                if len(target) != 6:
                    record["notes"].append("第 %d 个扫描位的关节角不是 6 个，跳过"
                                           % (view_index + 1))
                    continue
                if self._pose_seen(target, seen_poses):
                    record["notes"].append("第 %d 个扫描位跟已经看过的位姿几乎相同，跳过"
                                           % (view_index + 1))
                    continue
                label = ("2-look-scan-%d" % view_index if anchor_index == 0
                         else "2-look-anchor%d-scan-%d" % (anchor_index, view_index))
                self.servo(target)
                moved = True
                seen_poses.append(target)
                look = self._look(label)
                result["stages"]["look_photo"] = look.get("report")
                snap_look_joints()
                attempt = {
                    "anchor_index": anchor_index,
                    "index": view_index,
                    "joints_deg": [round(v, 2) for v in target],
                    "aim_m": [round(float(v), 4) for v in view.get("aim_m") or []],
                    "distance_m": (None if view.get("distance_m") is None
                                   else round(float(view["distance_m"]), 4)),
                    "view_angle_deg": (None if view.get("view_angle_deg") is None
                                       else round(float(view["view_angle_deg"]), 2)),
                    "deviation_deg": (None if view.get("deviation_deg") is None
                                      else round(float(view["deviation_deg"]), 2)),
                    "ok": bool(look["ok"])}
                if not look["ok"]:
                    attempt["why"] = look.get("why")
                    attempt["retryable"] = bool(look.get("retryable"))
                record["attempts"].append(attempt)
                attempts.append(attempt)
                if look["ok"]:
                    winning_anchor, winning_view = anchor_index, view_index
                    record["winning_view"] = view_index
                    log.info("%s 的第 %d 个观察位看准了（相对该锚点关节偏移 %.1f°）",
                             anchor_name, view_index + 1, attempt["deviation_deg"] or 0.0)
                    break
                if not look.get("retryable"):
                    # 连不上/服务报错：再换下一个观察位也是同样的下场，别白走。
                    attempt["stopped"] = "失败不可重试，停止扫描"
                    stop_reason = ("%s 的第 %d 个观察位失败且不可重试（kind=%s），停止扫描"
                                   % (anchor_name, view_index + 1, look.get("kind")))
                    log.info("%s 的第 %d 个观察位失败且不可重试（kind=%s），停止扫描",
                             anchor_name, view_index + 1, look.get("kind"))
                    break
            if winning_anchor is not None or stop_reason:
                break
        if winning_anchor is None and moved:
            # 全看不准：回标称位。**不留在某个随手选的观察位**——人过来看现场时，
            # 臂停在标称拍照位才跟"这一轮没成"对得上，下一轮也从同一处起手。
            self.servo(self._photo_pose())

        # 顶层 notes 把各锚点/规划器的说明都带出来（只看 notes 也能查原因）。
        for record in anchors:
            for note in record["notes"]:
                notes.append(note if record["anchor_index"] == 0
                             else "%s：%s" % (record["name"], note))
        if stop_reason:
            notes.append(stop_reason)
        # 兼容老读法：planned / attempts / winning_index 取"赢的那个锚点"那一份（没赢就取
        # 最后一个真的规划过的锚点）；多锚点的完整记录在 anchors 里。
        focus = anchors[0]
        if winning_anchor is not None:
            focus = anchors[winning_anchor]
        else:
            for record in reversed(anchors):
                if record["planned"] or record["attempts"]:
                    focus = record
                    break
        triggered = bool(attempts) or any(record["look"] is not None for record in anchors)
        return look, stage(
            triggered=triggered, surface_z_m=surface_z, anchors=anchors,
            winning_anchor=winning_anchor,
            winning_index=(winning_view if winning_view is not None and winning_view >= 0
                           else None),
            planned=list(focus["planned"]), notes=notes, attempts=attempts,
            returned_to_nominal=(winning_anchor is None and moved))

    def _walk_to_hover(self, hover_tip, look: dict) -> dict:
        """走到悬停位（物体上方 10 厘米），返回实到的关节角。

        工具轴**严格垂直于承托面**（tilt_limit_deg=0）：悬停是下爪那一步的起点，
        也是抬起要回的坐标，姿势得跟抓取一致——放斜的悬停在下一步再转回来，就是
        那类"先转过去再转回来"的多余动作。

        绕轴的闭合方向也照**抓取计划那条规则**来（物体长轴在平面上的垂线），并且用
        同一眼的检测结果：悬停按别的方向站（比如随手取 x 轴），下一步规划出来的抓取
        姿态就会差出几十度，下爪时腕子要在物体上方转一大圈——实测差 110°，被关节
        移动闸挡住，那一轮就是"悬停到位了、下爪一步不走"。
        """
        normal = np.asarray(look["plane"]["normal"], dtype=float)
        # 长轴和闭合方向由**视觉服务**算好带回来（它在相机系里从框和角度推出来，_look 已经
        # 转到基座系）。本文件不再从像素推——见视觉服务 grasp_geometry。
        long_axis = np.asarray(look["long_axis"], dtype=float)
        rough = {"ok": True, "problems": [], "plane": look["plane"],
                 "grasp_tip": np.asarray(hover_tip, dtype=float).tolist(),
                 "approach": (-normal).tolist(),
                 "closing": np.asarray(look["closing"], dtype=float).tolist()}
        # 站得住还不够：从这儿还得下得去。悬停解和抓取解落在**两支不同臂型**上时
        # （肘上/肘下那种差别），下爪要整条臂重排上百毫米——坡度 48.7° 且斜坡推到边
        # 的摆法下实测 J2 要转 96°、J6 109°，被关节移动闸挡住，那一轮就是"悬停到位、
        # 下爪一步不走"。所以顺手把估计的抓取点也解一遍，解不下去就换个闭合方向重站。
        estimated_grasp = None
        rotation_est = None
        object_height = float((np.asarray(look["top_point"], dtype=float)
                               - np.asarray(look["plane"]["point"], dtype=float)) @ normal)
        descent = grasp.solve_descent(max(object_height, 0.0),
                                      float(self.calib.gripper["clamp_length_mm"]) / 1000.0)
        if descent["ok"] and descent["height_m"] is not None:
            estimated_grasp = np.asarray(look["top_point"], dtype=float) \
                + (descent["height_m"] - object_height) * normal
            rotation_est = grasp.rotation_from_axes(-normal, np.asarray(rough["closing"]))

        def descends_from(solved) -> bool:
            """从这副悬停姿势出发，下爪那一步做不做得到。"""
            if estimated_grasp is None or not hasattr(self.calib, "seed"):
                return True                      # 估不出抓取点 / 真机那份没这道闸，不预判
            self.calib.seed(solved["joints_deg"])
            down = self.calib.solve(estimated_grasp, rotation_est, n_starts=8,
                                    max_joint_move_deg=DESCEND_MAX_JOINT_MOVE_DEG,
                                settled_m=grasp.descend_settled_m(self.calib.gripper,
                                                                  look.get("footprint_width")))
            return float(down["position_error_m"]) <= grasp.descend_settled_m(
                self.calib.gripper, look.get("footprint_width"))

        # n_starts=8：IK 换成"早认输 + 铺种子"之后，单次尝试放弃了早，预算得跟着涨——
        # 实测工作区 120 个目标，预算 3/4/8 分别到位 87/93/116 个，8 是拐点。
        # 悬停的约束只留一条：**从这儿能一步下到抓取点**（validate=descends_from）。
        # 其余全放开，理由是这个函数自己的注释——"悬停只是路过，下爪的目标是按实到的
        # 爪尖算的"：下爪那一步是从臂实到的关节角重新解 IK，悬停偏几毫米会被它吃掉，
        # 横向/沿轴那两条判据在悬停这儿是冗余的（在抓取那一步才有意义）。
        # 接近方向也从"只许严格垂直"放开到和抓取同档：这个位置上竖直朝下的姿势
        # 关节够不到（实测差 29 mm、姿态还歪了 7°），放斜 10° 就精确到位 1.09 mm——
        # 而姿势一旦成立，descends_from 会保证下爪那一步仍然做得成。
        reach = motion.plan_reach(self.calib, rough,
                                  tilt_limit_deg=HOVER_OBLIQUITY_DEG,
                                  n_starts=8, step_deg=45.0, validate=descends_from,
                                  closing_tol_deg=HOVER_CLOSING_TOL_DEG,
                                  lateral_tol_m=HOVER_LATERAL_TOL_M,
                                  along_tol_m=HOVER_ALONG_TOL_M)
        if not reach["ok"]:
            if reach.get("validate_rejected"):
                # 站得住、但"从这儿下不去"的候选被加试挡掉了，而且没有更好的。这不是
                # "解不出来"，是**这个摆法下没有既能站住又能直落下去的姿态**：能站住的
                # 那几支，下爪都要整条臂换臂型（超过关节移动闸的 70°）；能下爪的那支，
                # 又站不到悬停点上。实测坡度 48.7° + 斜坡推到边时就是这样。
                return {"ok": False, "failure_kind": "out_of_reach",
                        "why": "方块正上方 10 厘米处站着容易，但站住之后没法规规矩矩直落下去——"
                               "站得住的几支臂型，下爪都要整条臂重新展开（要转 100° 以上，超过 70° 的"
                               "关节移动闸）；能一步下的那支又到不了悬停点。多半是方块太靠外或坡太陡："
                               "把斜坡往回拉一点、或把坡度调小点再试。"}
            return {"ok": False, "failure_kind": "out_of_reach",
                    "why": reach.get("why", "")}
        target_rotation = np.asarray(reach["solved"]["rotation"], dtype=float)
        walked = motion.move_tip(self.calib, self.servo, rough["grasp_tip"],
                                 target_rotation,
                                 rounds=3, n_starts=8, tolerance_m=HOVER_TOLERANCE_M)
        actual_joints = list(self.client.state()["joints_deg"])
        actual_rotation = self.calib.flange(actual_joints)[:3, :3]
        # 记下**这一站站成的姿态**、它跟这一站目标姿态差多少度（伺服残差），以及每一刀
        # 的完整过程。下爪那一步用的是**另一套**目标姿态（规划解出来的那套），两者差多少
        # 见 run() 里记的 hover_vs_target_deg——那个数才是"悬停到位、下爪一步不走"的
        # 第一嫌疑（实测踩过，见本函数开头那段）。
        return {"ok": True, "tilt_deg": reach["tilt_deg"],
                "error_mm": (walked["error_m"] or 0.0) * 1000,
                "joints_deg": actual_joints,
                "target_rotation": target_rotation.tolist(),
                "actual_rotation": actual_rotation.tolist(),
                "orientation_gap_deg": orientation_gap_deg(actual_rotation, target_rotation),
                "history": walked["history"],
                "solve_debug": getattr(self.calib, "last_solve_debug", None)}

    def check(self) -> dict:
        """只"看"和"算"：拍一眼 → 检测 → 拟合承托面 → 解算抓取 → 搜可达解。**不动臂、不合爪。**

        「抓取测试」页上「检测一下」按钮走它。真抓之前先用当前画面看这一眼能不能过（框对不对、
        顶面量没量到、物体多高、够不够得着），省得每次都用一整轮实机动作去试。走的是 run()
        第 2、3 步**同一段代码**，所以它说能过、真跑就该能过；它说不通过，真跑也不用试了。
        """
        result = {"ok": False, "action": "check", "problems": [], "stages": {}}
        self._t0 = time.monotonic()
        look = self._look("0-check-look")
        result["stages"]["look_photo"] = look.get("report")     # 报告里与真抓一个口径
        if not look["ok"]:
            result["problems"].append("没看准：" + look["why"])
            return result
        plan = grasp.plan(look["top_point"], look["plane"], look["long_axis"],
                          look["closing"], look["footprint_width"], self.calib.gripper)
        result["stages"]["plan"] = {
            "object_height_mm": plan["object_height_m"] * 1000,
            "tip_height_mm": (plan["tip_height_m"] or 0.0) * 1000,
            "descent_mm": (plan["descent_m"] or 0.0) * 1000,
            "hover_tip_m": np.round(plan["hover_tip"], 4).tolist(),
            "grasp_tip_m": (None if plan["grasp_tip"] is None
                            else np.round(plan["grasp_tip"], 4).tolist()),
        }
        if not plan["ok"]:
            result["problems"].append("计划没通过：" + "；".join(plan["problems"]))
            return result
        reach = motion.plan_reach(self.calib, plan,
                                  tilt_limit_deg=grasp.MAX_OBLIQUITY_DEG, n_starts=8)
        result["stages"]["reach"] = {"ok": reach["ok"], "why": reach.get("why", ""),
                                     "tilt_deg": reach.get("tilt_deg"),
                                     "closing_offset_deg": reach.get("closing_offset_deg")}
        if not reach["ok"]:
            result["problems"].append("够不着：" + reach.get("why", ""))
            return result
        result["ok"] = True
        return result

    def run(self) -> dict:
        """执行一轮并在成功、失败、异常三种情况下都留现场证据。"""
        self._run_started = time.time()
        try:
            result = self._run_impl()
        except Exception as error:
            result = {"ok": False, "action": "grasp", "environment": self.scene.get("environment"),
                      "problems": [f"{type(error).__name__}: {error}"], "stages": {},
                      "exception": f"{type(error).__name__}: {error}"}
            log.exception("抓取流程异常")
        result.setdefault("diagnostics", {})
        result["diagnostics"].update({"started_unix": self._run_started,
                                       "timings": self._timings,
                                       "capture_files": sorted(p.name for p in self.out_dir.iterdir())
                                       if self.out_dir is not None else []})
        self._write_run_artifacts(result)
        return result

    def _run_impl(self) -> dict:
        """一整轮，按固定顺序走八步：

            1 去拍照位（固定姿势）
            2 拍照检测 → 找到物体 → 算出"物体上方 10 厘米"（悬停位）并记下这个坐标
              （看不准且是"这一眼没量出来"那类失败时，先换几个自动规划的观察位再看；
               见 _observe——找到即停，全都失败就回标称位并如实报）
            3 解算抓取（顶面、承托面、框都用第 2 步那一份）：计划不过关在这儿就停，
              臂一步都不走
            4 走到悬停位
            5 下爪
            6 合抓
            7 抬起 = 走回悬停位（用第 4 步记下的坐标）
            8 回位（用开头记下的坐标）

        看只在第 2 步看：标称拍照位那一眼（看不准时再在几个备用观察位各看一眼）。
        悬停位离物体太近、深度图上看不到物体，量不出东西来。
        后面每一步只读两样：话题上量出来的当前状态、第 2 步留下的那一份测量。留在手里
        的坐标有两个——回位那一个、悬停位那一个。
        """
        result = {"ok": False, "action": "grasp", "environment": None,
                  "problems": [], "stages": {}}
        self._t0 = time.monotonic()          # 阶段计时从这里起表
        target = self.scene["target"]
        # 回位要用的那个坐标（唯一从开头留到结尾的东西）
        initial_joints_deg = list(self.client.state()["joints_deg"])

        # ---- 1、2. 去拍照位看一眼：看清了就往下走；"这一眼没量出来"时才换几个
        #      自动规划的观察位再找（找到即停、全失败回标称位）。见 _observe ----
        look, scan_stage = self._observe(result)
        if scan_stage is not None:
            result["stages"]["observe_scan"] = scan_stage
        if not look["ok"]:
            attempts = (scan_stage or {}).get("attempts") or []
            if attempts:
                # 扫过：把"试了几个位"写进问题里，不然跟"只拍了一眼"混在一起，
                # 事后看不出到底换过角度没有。
                result["problems"].append(
                    "在拍照位和 %d 个备用观察位都没看准：%s" % (len(attempts), look["why"]))
            else:
                result["problems"].append("在拍照位没看准：" + look["why"])
            return result
        # 悬停位 = 物体上方 10 厘米，用**最后看准的那一眼**的量算（扫过就是扫描位那一份）。
        hover_tip = look["top_point"] + grasp.HOVER_M * look["plane"]["normal"]
        result["stages"]["hover_target_m"] = np.round(hover_tip, 4).tolist()

        # ---- 3. 解算抓取：顶面、承托面、框、相机位姿都用第 2 步那一份；**先算再动** ----
        # 悬停位不拍照：相机离物体只有 ~0.24 m（爪尖顶在物体上方 10 cm，相机在夹爪上、
        # 比爪尖更靠后），已在相机最近可测距离以内——物体在深度图上是块空洞，框里的有效
        # 深度只剩旁边的桌面（实测框内有效深度 0.8%、物体正面 0）。之前拿悬停位那一眼
        # 的框内深度中位数当顶面，量到的是桌面上的点，比承托面还低 11 mm，计划就卡在
        # "顶面落在平面之下、物体太矮"。拍照位离物体 60 cm，顶面、承托面、框都量得到，
        # 四个量同出一眼，也就不存在"两眼的取样时刻对不上"。
        #
        # 计划和解算都是纯数学（不碰硬件），所以放在走位**之前**把关：量错了就该在量错的
        # 那一眼就报出来。之前排在第 4 步（悬停之后），"物体太矮"要等臂白走一趟才报——
        # 现场实测就是这么发生的。
        plane_normal = np.asarray(look["plane"]["normal"], dtype=float)
        plan = grasp.plan(look["top_point"], look["plane"], look["long_axis"],
                          look["closing"], look["footprint_width"], self.calib.gripper)
        if not plan["ok"]:
            result["problems"].append("计划没通过：" + "；".join(plan["problems"]))
            return result
        # 抓取这一下要走满：候选够不着就换下一个（里面按"离首选多近"排序、过一个就停），
        # 预算是 8 的道理同悬停那步。
        reach = motion.plan_reach(self.calib, plan,
                                  tilt_limit_deg=grasp.MAX_OBLIQUITY_DEG, n_starts=8)
        result["stages"]["plan"] = {
            "object_height_mm": plan["object_height_m"] * 1000,
            "tip_height_mm": plan["tip_height_m"] * 1000,
            "descent_mm": plan["descent_m"] * 1000,
            "grasp_tip_m": np.round(plan["grasp_tip"], 4).tolist(),
            "closing_offset_deg": reach.get("closing_offset_deg"),
            # 闭合方向上量出来的宽度。合爪那道"有没有夹到物体"的闸用它当基准——
            # 名义宽度（half_size 的 2×min）在物体歪着躺时会差好几毫米（见合爪那段）。
            "footprint_width_mm": (None if plan.get("footprint_width_m") is None
                                   else plan["footprint_width_m"] * 1000),
        }
        if not reach["ok"]:
            result["problems"].append("够不着：" + reach.get("why", ""))
            result["failure_kind"] = "out_of_reach"
            return result
        rotation = np.asarray(reach["solved"]["rotation"], dtype=float)

        # ---- 4. 走到悬停位，把实到的关节角记下来（第 7 步抬回这里） ----
        hover = self._walk_to_hover(hover_tip, look)
        result["stages"]["hover"] = hover
        if not hover["ok"]:
            result["problems"].append("悬停这一步没走成：" + hover.get("why", ""))
            if hover.get("failure_kind"):
                result["failure_kind"] = hover["failure_kind"]
            return result
        hover_joints_deg = list(hover["joints_deg"])
        log.info("悬停到位：误差 %.1f mm，站成的姿态与这一站的目标差 %.1f°",
                 hover.get("error_mm") or 0.0, hover.get("orientation_gap_deg") or 0.0)

        # （夹爪在第 1 步去拍照位时就张开了，中途没有合过，这里不再重复开合：
        # 一次动作只做一次。）

        # ---- 5. 下爪 ----
        # 悬停位本身就是下爪起点。从话题读回当前实到关节角作为 IK 起点，直接解抓取点；
        # 不再另外造一个高位途径点，否则同一段下降会被拆成两次独立 IK，容易在中间换分支。
        before_descend = self.client.state()["joints_deg"]
        if hasattr(self.calib, "seed"):
            self.calib.seed(before_descend)
        descend_settled = grasp.descend_settled_m(self.calib.gripper, plan.get("footprint_width_m"))
        descend_kwargs = ({"max_joint_move_deg": DESCEND_MAX_JOINT_MOVE_DEG,
                           "settled_m": descend_settled}
                          if hasattr(self.calib, "seed") else {})
        # 先留一份这一步要用的目标姿态：下面为了后续步骤会把 rotation 换成**实到的**
        # 姿态，而日志里要拿"目标"去跟悬停站成的姿态比（见 hover_vs_target_deg）。
        descend_target_rotation = np.asarray(rotation, dtype=float).copy()
        down = motion.move_tip(self.calib, self.servo, plan["grasp_tip"], rotation,
                               rounds=2, n_starts=8, **descend_kwargs)
        descend_debug = getattr(self.calib, "last_solve_debug", None)
        state = self.client.state()
        # 后续每一步（抬起）都用**实到的那副姿态**当目标：夹爪绕工具轴 180°
        # 是对称的，下爪这一步可能在两族等价解里挑了另一族，后面再拿原来那族去要求它，
        # 就会解出"同一个夹法、整条臂绕一大圈"的动作（现场：J2 要转 110°，被闸挡住）。
        rotation = np.asarray(self.calib.flange(state["joints_deg"])[:3, :3], dtype=float)
        tip_now = np.asarray(state["tip"], dtype=float)
        result["stages"]["descend"] = {
            "rounds": down["rounds"], "error_mm": (down["error_m"] or 0.0) * 1000,
            "tip_m": np.round(tip_now, 4).tolist(),
            "tip_height_planned_mm": plan["tip_height_m"] * 1000,
            "tip_height_actual_mm": float((tip_now - np.asarray(look["plane"]["point"], dtype=float))
                                          @ plane_normal) * 1000,
            "tip_vs_plan_mm": float(np.linalg.norm(tip_now - np.asarray(plan["grasp_tip"]))) * 1000,
            # 出问题时最要紧的两组数：**命令**的关节角是多少、求解器内部试了哪些解。
            # 之前只记了实到的关节角，"命令走了一支、实到的是另一支"这种事故查不出来
            # （现场：爪尖差 21.5 mm、合爪开口被挤到 29.6 mm，事后不知道命令发的是什么）。
            "commanded_joints_deg": [round(float(v), 2)
                                     for v in down["solved"]["joints_deg"]] if down.get("solved") else None,
            # 主解/等价解两族各自的尝试表（种子、误差、关节移动）与最终采用了哪一族。
            # 少了它，"下爪没到位"只能看出差多少毫米，看不出是"根本没解出来"还是
            # "解出来了但超出关节移动闸"——这两种的原因和改法完全不同（真踩过）。
            "descend_ik": descend_debug,
            # 悬停站成的姿态 vs 下爪这一步的目标姿态，差多少度。这两个姿态是**分别算
            # 出来的**（悬停那步自己解一套、下爪用规划那套），差得多就得让腕子在物体
            # 上方先转过去；转的幅度顶到关节移动闸上，现象正是"悬停到位、下爪一步不走"
            # （本文件 _walk_to_hover 开头记过这个坑）。之前日志里没有这个角度。
            "hover_vs_target_deg": orientation_gap_deg(
                self.calib.flange(hover_joints_deg)[:3, :3], descend_target_rotation),
            # 这一刀每一轮的完整记录（目标/命令/实到的爪尖、每轮的 IK 误差与关节移动）。
            # move_tip 一直在记，只是之前没往结果里放。
            "history": down["history"],
        }
        log.info("下爪：悬停姿态与目标姿态差 %.1f°，实到爪尖差 %.1f mm，求解分支=%s",
                 result["stages"]["descend"]["hover_vs_target_deg"],
                 (down["error_m"] or 0.0) * 1000,
                 (descend_debug or {}).get("selected", "无记录"))

        if (down["error_m"] or 0.0) > DESCEND_ABORT_M:
            # 两种"没到位"要分开说，不然会把人带偏（实测被带偏过一整轮）：
            #   求解器一支都没收（keep_fallback）→ **它一步没动**，是"从这儿够不太到"，
            #   跟碰撞无关；报"被顶住了"会让人去查碰撞，方向全错。
            #   收下了、也走了、但停在半路 → 那才可能是被东西挡住。
            if (descend_debug or {}).get("selected") == "keep_fallback":
                # Same "out of reach" family as the plan_reach failure above: the
                # arm cannot reach the point from here, so backing the base up a
                # little is a legitimate retry.
                result["failure_kind"] = "out_of_reach"
                result["problems"].append(
                    "下爪没到位（爪尖差 %.1f mm，判据 %.0f mm）：**从悬停位解不到那个抓取点**"
                    "（最接近的解也差这么多），所以一步没动——不是被东西挡住。"
                    "没有合爪，这一轮作废。让这个点够得着：把物体往臂这边挪一点，"
                    "或者让狗站近一点。"
                    % ((down["error_m"] or 0.0) * 1000, DESCEND_ABORT_M * 1000))
            else:
                result["problems"].append(
                    "下爪没走到位（爪尖差 %.1f mm，判据 %.0f mm）：解出来了、也走了，"
                    "但停在半路——多半是被别的东西挡住了（越不过去挤过去会把物体推走）。"
                    "没有合爪，这一轮作废。"
                    % ((down["error_m"] or 0.0) * 1000, DESCEND_ABORT_M * 1000))
            return result

        # ---- 5.5 横向校正已删 ----
        # 原来是下爪之后再解一次 IK、把爪尖挪到方块中心正上方（退出判据 0.3 mm）。删掉的
        # 理由：这台机器一次移动的稳态误差就是毫米级，0.3 mm 永远达不到；而真机日志实测
        # 它带来的是"原地不动"的解 + 一条 4 秒白走的移动（两次，共 8 秒）。横向残差由下爪
        # 那两刀一起吃（第二刀就是按实到爪尖的残差补的），补不齐也直接合爪。
        # 记下停住时的关节角：光看爪尖位置算不出两根指头在哪儿，得拿姿态回放模型
        # 才知道它们是不是真的骑在方块两侧（还是有一根压在顶面上）。
        result["stages"]["descend"]["joints_deg"] = [
            round(v, 2) for v in self.client.state()["joints_deg"]]

        # ---- 6. 合爪：远段大步、接近接触后小步逼近，顶住之后按实测开口加压捏住 ----
        # 判据是"开口被顶住不再跟着命令走"而不是"没合到命令值"：小步之下被顶住
        # 时实际开口只比命令高半档。但顶住不等于捏住——判停时的捏力只有 ~0.1 N，
        # 抬不起 0.47 N 的方块（实测：短抬起时方块纹丝不动留在斜面）。所以顶住之后
        # 还要加压到 CLOSE_SQUEEZE_N（见下面那段；力 = kp × 干涉量）。
        # 真机同理：接触判断靠开口停止变化，捏住靠接触后的有限加压，不追命令值到底。
        steps = []
        contact = None
        previous = None
        for opening in CLOSE_COARSE_OPENINGS_M:
            actual = self.client.gripper(opening)
            steps.append({"commanded_mm": opening * 1000, "actual_mm": actual * 1000})
            previous = actual
        opening = CLOSE_COARSE_OPENINGS_M[-1]
        while opening > CLOSE_FINE_LIMIT_M and contact is None:
            opening -= CLOSE_FINE_STEP_M
            # may_be_blocked：细步就是逼近接触的，**越接近动得越少**——最后那一步实测
            # 只动了 0.17 mm，低于"动过"的门槛（0.2 mm），于是判据永远不成立、耗满
            # 6 秒超时。合爪 9.6 秒里这一个坑占了 6 秒。
            actual = self.client.gripper(opening, may_be_blocked=True)
            steps.append({"commanded_mm": opening * 1000, "actual_mm": actual * 1000})
            if previous is not None and actual >= previous - CLOSE_BLOCKED_TOL_M:
                contact = steps[-1]
            else:
                previous = actual
        # 顶住 ≠ 夹住：开口停在比物体宽出一大截的地方，说明指头根本没贴到物体上
        # （骑在方块边上、或先撞上别的），这时挤压是白压、抬起是抬空手——现场看着
        # 就是"夹具没闭合"。尺寸对不上就到此为止，如实报出来。
        # 闭合方向上物体的**实际**宽度：优先用规划那一步量出来的那个（把检测框的四角打到
        # 承托面上、量闭合方向的跨度，见视觉服务 grasp_geometry 的 image_geometry），
        # 量不到才退回名义宽度。
        #
        # 为什么不能用名义宽度：它默认"物体与承托面平行"。方块平躺在斜坡上时这正好成立，
        # 所以一直没暴露；物体一歪前提就破了——实测方块倾 11.3° 时，闭合方向的真实宽度是
        # **46.6 mm**，名义值只有 40 mm，于是**一次正确的夹持被判成"宽出 5.3 mm、没夹到
        # 物体上"整轮作废**（现场就是这么抓失败的）。而那个估计是偏大的（框打到平面上会
        # 比实物略长），方向上正好保守。
        footprint_m = plan.get("footprint_width_m")
        if footprint_m:
            held_width_mm = float(footprint_m) * 1000.0
        else:
            held_width_mm = 2.0 * float(min(target["half_size"][0],
                                            target["half_size"][1])) * 1000.0
        if contact is not None and contact["actual_mm"] - held_width_mm > CLOSE_CONTACT_MAX_GAP_MM:
            result["problems"].append(
                "夹爪在 %.1f mm 就被顶住了，而物体只有 %.0f mm 宽（宽出 %.1f mm）——没夹到物体上。"
                "多半是指头骑到了方块边上（下爪位置没对准），或者承托面/别的东西挡住了指头。"
                "这一轮到此为止，不抬空手。"
                % (contact["actual_mm"], held_width_mm, contact["actual_mm"] - held_width_mm))
            result["stages"]["close"] = {"steps": steps, "contact": contact,
                                         "final_opening_mm": contact["actual_mm"],
                                         "contact_wider_than_object_mm":
                                             contact["actual_mm"] - held_width_mm}
            return result
        # 顶住之后加压捏住：命令 = **当前实际开口 − 干涉量**，反复压到指头真落在物体上。
        #
        # 为什么盯"实际"而不是把命令往下数：接触可能是在离物体还有几毫米的地方顶住的
        # （指头别住了角、或先碰到别的东西），那一刻按命令数出来的干涉是虚的。指头后来
        # 落到物体上时，力只剩"物体宽度与命令之差"。实测坡度 0°：卡在 44.9 mm 判成接触、
        # 命令压到 38.5，指头后来落到方块上（40.8），捏力只有 0.45 N——回位途中方块滑掉了；
        # 对照坡度 48.7°：接触就在 41.8（贴着宽度），同样压到 38.5，指头真在方块上，
        # 1.2 N，一路拿得住。所以每次按实测值重压；指头落到位的那一轮，力自然就是
        # kp × 干涉量。压不动就说明指间是空的，rounds 用完了就如实记下来。
        interference_mm = CLOSE_SQUEEZE_N / GRIPPER_KP * 1000.0
        for _ in range(SQUEEZE_ROUNDS):
            if contact is None:
                break
            settled_mm = steps[-1]["actual_mm"]
            if settled_mm - steps[-1]["commanded_mm"] >= interference_mm:
                break                                # 已经压到位，不用再压
            command_mm = settled_mm - interference_mm
            # may_be_blocked：这几步就是要压在方块上，指头被顶住不动是正常的
            actual = self.client.gripper(command_mm / 1000.0, may_be_blocked=True)
            steps.append({"commanded_mm": command_mm, "actual_mm": actual * 1000})
        # 压完之后再等一次"真的不动了"，然后才允许抬起：合爪没闭拢就走抬起，指头会在
        # 抬起途中才夹到方块，夹到那一下随即滑脱（现场症状：下爪时夹具没闭拢、边抬边闭）。
        # 等停之后读数才是稳的，捏力也才估得准。
        if contact is not None:
            settled_mm = self.client.gripper_settle(steps[-1]["commanded_mm"] / 1000.0)
            steps.append({"commanded_mm": steps[-1]["commanded_mm"],
                          "actual_mm": settled_mm * 1000, "settled": True})
        # 捏力按最后一次"实际 − 命令"估（kp × 干涉量，两指合计；顶住时位置伺服的力就是它）。
        # 实测 0.34 N 抬不起方块、1.68 N 抬得起——这个数直接决定拿不拿得住，
        # 落进结果里，下次"抬起来半路掉了"不用再回放。
        pinch_n = GRIPPER_KP * max(0.0, steps[-1]["actual_mm"] - steps[-1]["commanded_mm"]) / 1000.0
        result["stages"]["close"] = {"steps": steps, "contact": contact,
                                     "final_opening_mm": steps[-1]["actual_mm"],
                                     "pinch_n": pinch_n,
                                     # 接触时开口比物体宽度大出多少：大了就是根本没贴上物体
                                     # （坡度 0° 那轮差 4.9 mm，抬到半路掉了）
                                     "contact_wider_than_object_mm": (
                                         (contact or {}).get("actual_mm", 0.0)
                                         - 2.0 * float(min(target["half_size"][0],
                                                           target["half_size"][1])) * 1000.0)
                                     if contact else None}

        # ---- 7. 抬起 = 走回悬停位（用第 3 步记下的那组关节角走关节空间） ----
        # 关节空间直走：同一个位置、同一段直线，腕子不会为了换一支解绕远。悬停位是
        # 第 3 步实到的那副姿势，所以这一趟就是把夹着的方块沿原路抬回去。
        lift_state = self.client.state()
        lift_joints = self.servo(hover_joints_deg)
        state = self.client.state()
        lift_tip = np.asarray(state["tip"], dtype=float)
        # 判据用**关节角**，不用爪尖距离：这一步是关节空间直走，能做到的精度就是伺服的
        # 重复性（到位判据 1.0°，实测残差 ≤0.95°/关节）；换算成爪尖距离会被臂长放大到
        # 五六毫米（1° × 0.35 m），拿闭环解 IK 的 1.5 mm 去卡它只会一路假报警——
        # 方块明明夹着、也抬起来了，却报"没回到悬停位"。爪尖距离照记，当参考。
        lift_error_mm = float(np.linalg.norm(lift_tip - np.asarray(hover_tip, dtype=float)))
        lift_error_deg = float(np.max(np.abs(
            np.asarray(lift_joints, dtype=float) - np.asarray(hover_joints_deg, dtype=float))))
        # 夹住要有两条证据：指间卡着东西，且方块真的跟着升起来了（话题位姿沿法线
        # 升高 > 5 mm）。光看开口会误报——方块挤脱了，指间一样有空隙。
        # 开口的门槛按**物体宽度**算：取两个水平半宽里小的那个，因为细长物体是横着
        # 夹的（用大的那个会永远判不过）。
        # "夹住了"的证据：有物体位姿源（仿真）时看方块有没有跟着升起来；真机没有那个源，
        # 客户端可以自己给判据（client.holding()，真机那份是"指头没合到命令位"）——
        # 给了就优先用它，返回 None 表示它也说不准、退回原判据。
        holding = getattr(self.client, "holding", None)
        holding_now = None
        if holding is not None:
            try:
                holding_now = holding()
            except Exception as error:                    # noqa: BLE001 —— 判据坏了不该拖垮整轮
                result["problems"].append("夹住判据出错（%s: %s），退回原判据"
                                          % (type(error).__name__, error))
        result["stages"]["lift"] = {
            "target_m": np.round(hover_tip, 4).tolist(),
            "tip_m": np.round(lift_tip, 4).tolist(),
            "error_mm": lift_error_mm * 1000,                     # 参考：爪尖差多少
            "error_deg": lift_error_deg,                          # 判据：关节角差多少
            "joints_deg": [round(float(v), 2) for v in lift_joints],
            # 判据只有一条：client.holding()（按"指头没合到命令位"判断）。原来还有一条
            # "物体跟着升起来了"，那要物体位姿源——真机没有，删了。
            "grabbed": bool(holding_now),
            "holding_evidence": "client.holding()" if holding_now is not None else "物体位姿",

            "gripper_opening_mm": state["gripper_m"] * 1000,
        }
        if lift_error_deg > LIFT_MAX_ANGLE_ERROR_DEG:
            result["problems"].append(
                "抬起没回到悬停位（关节角差 %.2f°，判据 %.1f°）：夹着方块走关节空间都偏这么多。"
                % (lift_error_deg, LIFT_MAX_ANGLE_ERROR_DEG))

        # ---- 8. 回位：送回 home（全零，方块仍握在手里） ----
        # 原来送回"抓取开始时的姿态"；起手若在拍照/悬停位，夹着东西举在外面重心不稳。
        # 统一回全零的 home 位，把负载收回机体附近。仍走关节空间（client.move_joints
        # 3 度一步插值），避免腕部大角度直线路径把指间的物体扫出去。
        home_target = [0.0] * 6
        home_joints = np.asarray(self.servo(home_target), dtype=float)
        after = self.client.state()
        # 方块还在不在手里，看它离**爪尖**多远。只看夹爪开口会误报——方块挤脱了，
        # 指间一样留着空隙（第 7 步的 grabbed 就是为这个才要两条证据）。
        still_held = bool(holding_now)
        result["stages"]["return_home"] = {
            "target_deg": [round(float(v), 2) for v in home_target],
            "actual_deg": [round(v, 2) for v in home_joints],
            "error_deg": float(np.max(np.abs(
                home_joints - np.asarray(home_target, dtype=float)))),
            "still_held": still_held,
            "gripper_opening_mm": after["gripper_m"] * 1000,
        }
        if result["stages"]["lift"]["grabbed"] and not still_held:
            result["problems"].append("回初始位途中方块掉了（client.holding() 说不在了）")

        # ---- 一致性：两路独立来源对同一物理量 ----
        result["consistency"] = self._consistency(state, look)
        # 成不成，**只看方块有没有真的进到指间并离开斜面**——那才是这件事的目标。
        # 下爪那一步的收敛判据（爪尖到目标 ≤1.5 mm）是精度指标，报出去，
        # 不拿来否决整轮：真偏到抓不着时，夹住那条本身就会是 False。
        grabbed = result["stages"]["lift"]["grabbed"]
        result["ok"] = bool(grabbed)
        if not result["ok"]:
            result["problems"].append("这一轮没抓起来（下爪 %s / 夹住 %s）"
                                      % (down["ok"], grabbed))
        elif not down["ok"]:
            result["problems"].append(
                "方块夹起来了，但下爪那一步没走到计划位（差 %.1f mm，判据 %.1f mm）"
                "——这一轮动作不干净，结果仍算夹住。"
                % ((down["error_m"] or 0.0) * 1000, descend_settled * 1000))
        self._write_run_artifacts(result)
        return result

    # ---------- 内部 ----------

    def _nearest_roll_equivalent(self, target_joints_deg) -> list:
        """回位的目标角里，把腕子的滚转（J6）换成"离现在最近的那一支"。

        这副夹爪绕接近轴是 180° 对称的：滚 180° 是**同一个夹法**（两指互换），所以 J6 有
        一族等价目标角（±180° 的整数倍）。取近的那支，回程里腕子就少转一整圈——现场：
        抓到手之后夹着方块转了两圈半（起手 J6=-126°、抓取后 J6=+154°，近的那边 -206°
        出了 ±179.9° 的限位，只能绕远 280°），方块差点被甩掉。
        """
        target = np.asarray(target_joints_deg, dtype=float).copy()
        if not hasattr(self.calib, "joint_limits_deg"):
            return target.tolist()               # 真机那份拿不到限位，原样交回
        limits = self.calib.joint_limits_deg()
        current = np.asarray(self.client.state()["joints_deg"], dtype=float)
        best = target
        for turn in (-2, -1, 1, 2):
            candidate = target.copy()
            candidate[5] += 180.0 * turn
            if any(candidate[i] < limits[i][0] or candidate[i] > limits[i][1]
                   for i in range(len(candidate))):
                continue
            if float(np.max(np.abs(candidate - current))) < float(np.max(np.abs(best - current))):
                best = candidate
        return best.tolist()

    def _consistency(self, state, look: dict) -> dict:
        """两路独立来源对同一物理量：**法兰位姿**。

        - 一路：/arm/end_pose（臂自己报的法兰位姿）加 tip_in_flange 推出来。
        - 另一路：/arm/joint_states_single 的关节角，按**标定链**做正解再推出来。

        两个话题独立、计算路径也不同。**这一项是真机上线时唯一能对上的对照**——它量的是
        "臂以为自己在哪" 和 "我们按标定算出来它在哪" 差多少。差值大就是在说：标定跟臂的
        名义运动学对不上（实测撞到过 17.6 mm，爪子整片偏出去、40 mm 的方块夹空）。

        物体那半删了：真机没有 /sim/objects，没有第二个来源可比。
        """
        flange_from_topic = self.to_base @ np.asarray(state["end_pose_world"], dtype=float)
        tip_from_end_pose = (flange_from_topic[:3, 3]
                             + flange_from_topic[:3, :3] @ self.calib.tip_in_flange)
        tip_from_joints = self.calib.tip(self.calib.flange(state["joints_deg"]))
        return {
            "end_pose_topic_m": np.round(state["end_pose_world"][:3, 3], 4).tolist(),
            "tip_from_end_pose_m": np.round(tip_from_end_pose, 4).tolist(),
            "tip_from_joints_m": np.round(tip_from_joints, 4).tolist(),
            "tip_sources_agree_mm": float(
                np.linalg.norm(tip_from_end_pose - tip_from_joints)) * 1000,
            "object_top_from_depth_m": np.round(np.asarray(look["top_point"]), 4).tolist(),
        }

    def _write_json(self, name: str, data: dict) -> None:
        if self.out_dir is None:
            return
        path = self.out_dir / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
                        encoding="utf-8")

    def _write_run_artifacts(self, result: dict) -> None:
        """写本轮的可复核证据：**只落结果 JSON**。

        原始帧不在这儿落了：帧在视觉服务那边（它每次请求都有自己的留档目录，
        原图、深度、结果都在 `log_dir` 下）。技能这边 `out_dir` 只放这一轮的
        state / 视觉诊断 / result，用于复盘"我们看到了什么、怎么决策的"。
        """
        if self.out_dir is None:
            return
        self._write_json("request_state.json", self._capture.get("state", {}))
        self._write_json("detector.json", self._capture.get("detector") or {})
        self._write_json("result.json", result)

    def _mark_stage(self, name: str) -> None:
        """打一个阶段时间点（"一轮的时间去哪了"看这个）。"""
        if self.out_dir is None:
            return
        if self._t0 is None:
            self._t0 = time.monotonic()
        elapsed = time.monotonic() - self._t0
        self._timings.append((name, round(elapsed, 2)))
        print("  [阶段] %-16s 累计 %6.1f s" % (name, elapsed), flush=True)

def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


