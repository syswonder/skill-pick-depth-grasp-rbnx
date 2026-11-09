# SPDX-License-Identifier: MulanPSL-2.0
"""pick.depth_grasp 的 atlas 桥：能力声明、生命周期、以及三个工具的编排。

本包**不带视觉算法**：检测、承托面拟合、顶面提取、"框和角度换算成三维"都在服务
`robonix/service/perception/grasp_geometry/locate` 那一侧跑（它自己订阅相机、取
最新一帧，结果在相机系）。技能这边经 `grasp_skill/vision_service.py` 调它，再把
相机系结果乘一次标定（`gripper_camera`）搬到基座系。**技能不订阅相机、不依赖宿主
HTTP 服务、也不碰 VLM 凭据。**

模块分工：

    pipeline.py      八步流程：拍照 → 检测 → 平面 → 计划 → 悬停 → 下爪 → 合爪 → 抬起
                     （"检测/平面"那两步是打给视觉服务的，本文件只取状态、搬坐标系）
    vision_service.py 视觉服务的 gRPC 客户端（atlas 解析 + Locate 调用 + 失败分类）
    client.py        ROS 侧的臂/夹爪（话题名由本文件经 atlas 解析后注入；**不含相机**）
    calib/           标定产物读取与 IK 链
    real_adapter.py  把部署配置拼成流水线认的 scene

**本包是真机专属**：没有仿真那条路（`calib_source: sim`、MuJoCo 模型、几何真值检测器
都删了）——仿真归 `tools/grasp_tilt`。

上游一律经 atlas 解析 robonix 原语契约，**不写死话题名**：解析不到就报错，不猜。
臂/夹爪走 `primitive/arm/{joint_states, joint_command, end_pose}`；夹爪不是单独的原语
——按 arm/joint_states 的契约说明，它是同一条 JointState 流里的一个具名关节。

生命周期按技能那一套（懒激活、可重入）：

    CMD_INIT       只解析配置，不碰 atlas / ROS / 模型
    CMD_ACTIVATE   解析原语端点与视觉服务端点 → 建标定/客户端 → 起 ROS 节点（可反复来）
    CMD_DEACTIVATE 反着拆一遍
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 算法件是从 tools/grasp_tilt 整份拷过来的，那边是平铺目录、用绝对导入（import motion
# 这种）。把包目录本身也放进 sys.path，拷过来的文件就一个字都不用改。
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from robonix_api import ATLAS, Err, Ok, Skill

from pick_mcp import (Observe_Request, Observe_Response, Pick_Request, Pick_Response,
                      PutDown_Request, PutDown_Response)

from . import real_adapter, vision_service
from .backoff import run_with_backoff
from .pipeline import Pipeline

log = logging.getLogger("depth_grasp")

NAMESPACE = "robonix/skill/pick"
D_PICK = NAMESPACE + "/pick"
D_PUTDOWN = NAMESPACE + "/put_down"
D_OBSERVE = NAMESPACE + "/observe"

# 与部署清单里 skill 实例的 name 对应；允许部署用环境变量改名（多实例时各用各的）。
skill = Skill(id=os.environ.get("ROBONIX_CAPABILITY_ID", "depth_grasp"),
              namespace=NAMESPACE)

# 原语契约 → Client 里那几个数据源的键。夹爪跟着臂走（见文件头）。**没有相机**：
# 视觉服务自己订阅相机，技能端只拿臂的状态与命令。
PRIMITIVES = {
    "joints": "robonix/primitive/arm/joint_states",
    "command": "robonix/primitive/arm/joint_command",
    "end_pose": "robonix/primitive/arm/end_pose",
}

# Pick.srv 里 grasp_pose 的坐标系：本技能所有位姿都在臂基座系（标定链就是按它建的）。
POSE_FRAME = "arm/base_link"


# ---------------------------------------------------------------- 配置

def _joints6(value, name: str):
    """把配置里的 6 个关节角（度）归一成 list[float]；None 原样返回。"""
    if value is None:
        return None
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError("%s 要是 6 个数（关节角，度），收到 %r" % (name, value))
    if len(values) != 6:
        raise ValueError("%s 要是 6 个数（关节角，度），收到 %d 个" % (name, len(values)))
    return values


@dataclass
class Config:
    """on_init 收到的 config 块。缺项都有默认值，坏值告警后回落。"""

    # "artifacts" 读标定产物目录（开发台旁边有 tools/ 时用）
    # "inline"    标定数直接写在下面的 calib: 里（机器人容器里没有 tools/）
    calib_source: str = "inline"
    calib_root: str = ""                    # calib_source=artifacts 时的产物根目录
    calib: dict = field(default_factory=dict)   # calib_source=inline 时的全部标定数
    out_dir: str = ""                       # 结果 JSON 落盘处；空则不落（帧在服务那边）
    # 多观察位扫描：标称拍照位"这一眼没量出来"时，换几个自动规划的观察位再看。
    # 键与默认值在 grasp_skill/observe_scan.py 的 SCAN_DEFAULTS 里。这里原样收着、
    # **不合并默认值**——合并留给规划那一步（observe_scan.merged_config）。
    observe_scan: dict = field(default_factory=dict)

    # ── 视觉服务（service/perception/grasp_geometry/locate）──
    # 多实例部署时用它去歧义（atlas 上同类服务不止一个）；空 = 要求恰好一个。
    vision_provider_id: str = ""
    vision_timeout_s: float = vision_service.DEFAULT_TIMEOUT_S

    # ── 下面几项只在 calib_source=artifacts（真机）时用 ──
    # 物体尺寸是**人量的**，用来算握着时开口该是多少。目标名字不在这儿：
    # 抓什么由调用方在 MCP 的 object_name / target 里说（必填）。
    object_size_mm: dict = field(default_factory=dict)
    table_height_m: float = 0.0             # 桌面在基座系的高度；只在诊断/扫描兜底里用
    # 够不着时的退避重试（见 grasp_skill/backoff.py）。
    backoff: dict = field(default_factory=dict)

    # ── put_down 的落点 ──
    # 6 个关节角（度）。没配就在当前位张开夹爪（不动臂）。
    put_down_joints_deg: list | None = None
    # 放完之后回的安全位（6 个关节角，度）。没配回全零 home。
    teach_safe_joints_deg: list | None = None

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Config":
        def text(key: str, default: str = "") -> str:
            value = cfg.get(key, default)
            return str(value).strip() if value is not None else default

        source = text("calib_source", "inline").lower()
        if source not in ("artifacts", "inline"):
            log.warning("不认识的 calib_source=%r，按 inline 走", source)
            source = "inline"
        calib = cfg.get("calib") or {}
        if not isinstance(calib, dict):
            log.warning("calib 不是字典，按空处理")
            calib = {}
        size = cfg.get("object_size_mm") or {}
        if not isinstance(size, dict):
            log.warning("object_size_mm 不是字典，按空处理")
            size = {}
        # 多观察位扫描的覆盖项：坏值只是"按默认走"，不该让整个技能起不来——它是一层
        # 补救措施，不是抓取的必要条件。
        observe_scan = cfg.get("observe_scan") or {}
        if not isinstance(observe_scan, dict):
            log.warning("observe_scan 不是字典，按空处理（用默认：开启、最多 4 个观察位）")
            observe_scan = {}
        backoff = cfg.get("backoff") or {}
        if not isinstance(backoff, dict):
            log.warning("backoff 不是字典，按空处理（用默认：开启、最多 1 次）")
            backoff = {}
        try:
            vision_timeout_s = float(cfg.get("vision_timeout_s")
                                     or vision_service.DEFAULT_TIMEOUT_S)
        except (TypeError, ValueError):
            log.warning("vision_timeout_s 不是数字，按默认 %.0f s",
                        vision_service.DEFAULT_TIMEOUT_S)
            vision_timeout_s = vision_service.DEFAULT_TIMEOUT_S
        if vision_timeout_s <= 0:
            log.warning("vision_timeout_s=%s 不是正数，按默认 %.0f s",
                        vision_timeout_s, vision_service.DEFAULT_TIMEOUT_S)
            vision_timeout_s = vision_service.DEFAULT_TIMEOUT_S
        return cls(
            calib_source=source,
            calib_root=text("calib_root"),
            calib=calib,
            out_dir=text("out_dir"),
            observe_scan=observe_scan,
            vision_provider_id=text("vision_provider_id"),
            vision_timeout_s=vision_timeout_s,
            object_size_mm={k: float(v) for k, v in size.items()},
            table_height_m=float(cfg.get("table_height_m") or 0.0),
            backoff=backoff,
            put_down_joints_deg=_joints6(cfg.get("put_down_joints_deg"),
                                         "put_down_joints_deg"),
            teach_safe_joints_deg=_joints6(cfg.get("teach_safe_joints_deg"),
                                           "teach_safe_joints_deg"),
        )


# ---------------------------------------------------------------- 运行时状态

@dataclass
class Runtime:
    config: Config | None = None
    client: object = None
    pipeline: object = None
    vision: object = None                   # grasp_skill.vision_service.VisionClient
    calib: object = None
    chassis: object = None                  # grasp_skill.chassis.ChassisClient
    facts: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = Runtime()


def _resolve_endpoint(contract_id: str, timeout_s: float = 60.0) -> tuple:
    """问 atlas 要一个原语的话题名与 QoS。解析不到就抛，不返回兜底值。

    为什么不退回默认话题名：默认值是仿真那一套，在别的部署里十有八九对不上，
    挂上去是**静默失联**——订阅建好了、一条消息不来，看起来像"算法坏了"。
    宁可这里报错。
    """
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            view = ATLAS.find_unique_capability(contract_id=contract_id, transport="ros2")
            channel = skill.connect_capability(view, contract_id, "ros2")
            endpoint = str(channel.endpoint or "")
            qos = getattr(getattr(channel, "params", None), "qos_profile", "") or "best_effort"
            try:
                channel.close()
            except Exception:                      # noqa: BLE001 —— 关不掉不影响解析结果
                pass
            if endpoint:
                return endpoint, str(qos)
            last_error = f"{contract_id}: atlas 给的端点为空"
        except Exception as error:                 # noqa: BLE001 —— 还没注册上就继续等
            last_error = f"{contract_id}: {error}"
        time.sleep(2.0)
    raise RuntimeError(f"atlas 上解析不到原语 {last_error}")


def _build_calib(config: Config):
    """按配置建标定对象：读标定产物目录，或者用清单里 inline 的数。

    没有仿真那一支了——本包是真机专属（仿真归 tools/grasp_tilt，它那份从 MuJoCo 模型
    现推）。
    """
    if config.calib_source == "artifacts":
        if not config.calib_root:
            raise RuntimeError("calib_source=artifacts 但没给 calib_root")
        # paths.py 里那些产物路径是 import 时算出来的，所以根目录只能在这之前用
        # 环境变量给——不是 import 完再设。
        os.environ["GRASP_DEV_CALIB_ROOT"] = config.calib_root
        from .calib.calib_io import Calibration
        return Calibration()
    if config.calib_source == "inline":
        if not config.calib:
            raise RuntimeError("calib_source=inline 但 config 里没有 calib 块")
        from .calib.calib_io import Calibration
        return Calibration.from_numbers(config.calib)
    raise RuntimeError("不认识的 calib_source=%r（真机这份认 artifacts / inline）"
                       % config.calib_source)


# ---------------------------------------------------------------- 生命周期

@skill.on_init
def on_init(cfg):
    """只解析配置。此刻 atlas 可能还没起齐、ROS 也不一定有，别在这儿碰它们。"""
    try:
        _state.config = Config.from_cfg(cfg or {})
    except Exception as error:                     # noqa: BLE001
        return Err(f"配置解析失败：{error}")
    log.info("depth_grasp 配置：calib=%s（抓什么由调用方的 target 指定）",
             _state.config.calib_source)
    return Ok()


@skill.on_activate
def on_activate():
    """把热资源建起来：原语端点、视觉服务端点、ROS 节点。可重入——先拆再建。"""
    if _state.pipeline is not None:
        return Ok()
    config = _state.config
    if config is None:
        return Err("还没收到 CMD_INIT 的配置")
    try:
        with _state.lock:
            # ROS 侧（rclpy）延迟到激活才导入：init 必须轻、且 `import
            # grasp_skill.atlas_bridge` 在没有 ROS 环境时也能过（codegen 冒烟）。
            from . import client as client_mod
            from . import rclpy_boot
            rclpy_boot.ensure_init()

            topics = {}
            for key, contract_id in PRIMITIVES.items():
                endpoint, qos = _resolve_endpoint(contract_id)
                topics[key] = endpoint
                log.info("原语 %s → %s [%s]", contract_id, endpoint, qos)
            calib = _build_calib(config)
            _state.calib = calib
            _state.facts = real_adapter.scene_from_config(config)
            # 视觉服务是依赖：解析不到就激活失败（不退回宿主 HTTP 服务）。
            vision = vision_service.open_client(skill, config)
            _state.vision = vision

            _state.client = client_mod.Client(calib, topics=topics)
            _state.pipeline = Pipeline(
                client=_state.client, calib=calib,
                servo=_state.client.move_joints,
                scene=_state.facts,
                vision=vision,
                observe_scan_config=config.observe_scan,
                # out_dir 空 = 不落盘（默认）。落也只落结果 JSON，不落原始帧——
                # 帧在视觉服务那边的请求留档里。
                out_dir=(Path(config.out_dir) if config.out_dir else None))
            # 够不着时的退避重试需要底盘（站/趴/前进）。解析不到底盘不算激活失败：
            # 没有底盘的部署照样能抓，只是不重试。
            if bool((config.backoff or {}).get("enabled", True)):
                from . import chassis as chassis_mod
                _state.chassis = chassis_mod.open_chassis(skill)
        log.info("depth_grasp 已激活")
        return Ok()
    except Exception as error:                     # noqa: BLE001 —— 激活失败要原样报出去
        log.exception("激活失败")
        _teardown()
        return Err(f"激活失败：{error}")


@skill.on_deactivate
def on_deactivate():
    _teardown()
    return Ok()


def _teardown() -> None:
    """拆干净。反复激活/停用时每次都走这里。"""
    with _state.lock:
        if _state.vision is not None:
            try:
                _state.vision.close()
            except Exception:                      # noqa: BLE001
                pass
        if _state.client is not None:
            try:
                _state.client.destroy_node()
            except Exception:                      # noqa: BLE001
                pass
        _state.vision = None
        _state.client = None
        _state.pipeline = None
        _state.calib = None
        _state.chassis = None
        _state.facts = {}
    try:
        from . import rclpy_boot
        rclpy_boot.ensure_shutdown()
    except Exception:                              # noqa: BLE001
        pass


# ---------------------------------------------------------------- 工具

def _json_default(value):
    """json.dumps 的兜底：numpy 标量/数组、Path 都能落进 detail_json。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _grasp_with_backoff(pipeline, config: Config | None) -> dict:
    """够不着（out_of_reach）时：回 home → 站立 → 前进一小段 → 趴下 → 再抓。"""
    backoff = (config.backoff if config is not None else {}) or {}

    def number(key: str, default: float) -> float:
        try:
            return float(backoff.get(key, default))
        except (TypeError, ValueError):
            log.warning("backoff.%s 不是数字，按默认 %s", key, default)
            return float(default)

    enabled = bool(backoff.get("enabled", True)) and _state.chassis is not None
    try:
        attempts = int(backoff.get("max_attempts", 1))
    except (TypeError, ValueError):
        attempts = 1
    max_attempts = max(0, attempts) if enabled else 0
    return run_with_backoff(
        pipeline, _state.client, _state.chassis,
        max_attempts=max_attempts,
        forward_m=max(0.0, number("forward_m", 0.10)),
        settle_s=max(0.0, number("settle_s", 0.5)),
        home_on_failure=bool(backoff.get("home_on_failure", enabled)),
    )


def _run(action: str, want_target: str = "") -> tuple:
    """跑一轮，返回 (success, stage, message, result)。

    result 是 pipeline 的整轮结果 dict（含 stages / problems / consistency），
    调用方拿它拼 detail_json 与 Pick.srv 的其它字段。
    """
    pipeline = _state.pipeline
    if pipeline is None:
        return False, "inactive", "技能还没激活（第一次调用前 executor 才会激活它）", {}
    if not (want_target or "").strip():
        # 空名字去问检测，等于让模型在画面里随便挑一个——那不叫"抓你说的那个"。
        return False, "target", "没给 target：要抓什么必须由调用方说（没有配置里的默认目标）", {}
    # 目标名字是唯一来源。名字前缀按 scene 的约定拼，pipeline 那边会去掉 "object_"。
    pipeline.scene["target"]["name"] = "object_%s" % want_target.strip()
    config = _state.config
    with _state.lock:                              # 一条臂同一时刻只能有一轮在跑
        stale = _state.client.stale_sources()
        if stale:
            # 节点在两次工具调用之间不 spin，所以每次进来 _seen 都会"过期"——
            # 哪怕话题一直在发。先 spin 最多 5 秒等数据真正到齐，再决定是否重建；
            # 实测直接 reset_links() 会把原来还能用的订阅拆成静默失效。
            import rclpy
            log.info("数据源不新鲜（%s），先 spin 等待数据", "、".join(stale))
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and _state.client.stale_sources():
                rclpy.spin_once(_state.client, timeout_sec=0.1)
            stale = _state.client.stale_sources()
        if stale:
            log.warning("数据源仍然不新鲜（%s），重建订阅", "、".join(stale))
            _state.client.reset_links()
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and _state.client.stale_sources():
                import rclpy
                rclpy.spin_once(_state.client, timeout_sec=0.1)
            still = _state.client.stale_sources()
            if still:
                return False, "data", \
                    "数据源断了（%s），重建订阅也没回来——话题那头没在发？" % "、".join(still), {}
        try:
            if action == "observe":
                result = pipeline.observe()
            else:
                result = _grasp_with_backoff(pipeline, config)
        except Exception as error:                 # noqa: BLE001 —— 一轮失败不该拖垮技能
            return False, "exception", f"{type(error).__name__}: {error}", {}

    problems = result.get("problems") or []
    if action == "observe":
        message = "看完一眼" if result.get("ok") else "；".join(problems) or "没看成"
        return bool(result.get("ok")), "observe", message, result
    if result.get("ok"):
        return True, "lift", "抓住了并抬起来了（回到 home 位，夹爪闭合，握着不释放）", result
    stage = str(result.get("stage") or _last_stage(result))
    reason = "；".join(problems) or "没成（原因见 detail_json）"
    return False, stage, reason, result


def _last_stage(result: dict) -> str:
    """没成的时候，报最后走到的那个阶段——比空着强。

    observe_scan 排在 look_photo **之后**：扫描失败时 stages 里 photo / look_photo /
    observe_scan 都在（都走过了），真正停下的那一步是扫描，reversed 先取到它才对。
    若排在 look_photo 前面，就会被误报成"停在 look_photo"——仿佛扫描没发生过。

    反过来，"跳过扫描"（triggered=false：配置关了、失败不可重试、没模块、估不出支撑面）
    也留了一份记录，但它没真扫过，报它同样误导——明确 false 就退回 look_photo。
    """
    stages = result.get("stages") or {}
    order = ("photo", "look_photo", "observe_scan", "plan", "hover", "descend", "close",
             "lift", "return_home")
    for name in reversed(order):
        if name not in stages:
            continue
        record = stages[name]
        if (name == "observe_scan" and isinstance(record, dict)
                and record.get("triggered") is False):
            continue
        return name
    return "unknown"


def _grasp_pose_from_result(result: dict):
    """本轮**实际执行**的抓取位姿，组装成 Pick.srv 的 `grasp_pose`。

    来源：下爪那一步停住时记下的实际关节角（`stages.descend.joints_deg`）→ 标定链
    正解得到法兰在基座系的位姿。取不到（没走到下爪就失败了）就给零位姿
    （位置全零、单位四元数），**不编一个看起来像抓过的位姿**。
    """
    from builtin_interfaces_mcp import Time
    from geometry_msgs_mcp import Point, Pose, PoseStamped, Quaternion
    from std_msgs_mcp import Header

    position = (0.0, 0.0, 0.0)
    quaternion = (0.0, 0.0, 0.0, 1.0)
    stages = result.get("stages") or {}
    descend = stages.get("descend") or {}
    joints = descend.get("joints_deg")
    if joints and _state.calib is not None:
        try:
            from scipy.spatial.transform import Rotation

            matrix = np.asarray(_state.calib.flange(joints), dtype=float)
            x, y, z, w = [float(v) for v in Rotation.from_matrix(matrix[:3, :3]).as_quat()]
            position = tuple(float(v) for v in matrix[:3, 3])
            quaternion = (x, y, z, w)
        except Exception as error:                 # noqa: BLE001 —— 取不到就给零位姿
            log.warning("从本轮结果里取抓取位姿失败，回零位姿：%s: %s",
                        type(error).__name__, error)
    return PoseStamped(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id=POSE_FRAME),
        pose=Pose(position=Point(*position), orientation=Quaternion(*quaternion)),
    )


def _gripper_width_from_result(result: dict) -> float:
    """抓取开口（米）：合爪那一步停住时的实际开口；取不到给 0。"""
    stages = result.get("stages") or {}
    close = stages.get("close") or {}
    value = close.get("final_opening_mm")
    if value is None:
        lift = stages.get("lift") or {}
        value = lift.get("gripper_opening_mm")
    try:
        return max(0.0, float(value)) / 1000.0
    except (TypeError, ValueError):
        return 0.0


@skill.mcp(D_PICK)
def pick(request: Pick_Request) -> Pick_Response:
    """抓取桌上的目标物体，抓住后保持握持并回到 home 位。

    **默认底盘保持趴下/坐姿，不要为了抓取调用底盘站立**；只有确实需要移动底盘时
    才站立，移动完再趴下。

    走到拍照位、调视觉服务量一眼、算出抓取点、下爪、合爪、抬起——成功时回 home 位、
    夹爪闭合、手里握着，不释放也不搬运。目标不在画面里、承托面量不出来、下爪没到位
    等情况不会硬合爪，会如实返回卡在哪一步以及原因（`message` 以阶段名开头，例如
    `descend: 下爪没到位…`）。

    如果只是**够不着**（plan_reach 失败，或下爪时从悬停位解不到抓取点）：先回 home、
    站立、前进 backoff.forward_m（默认 0.10 m）、趴下，再重新观察抓取，最多
    backoff.max_attempts 次（默认 1）。看不清/量不准/被挡住不重试。最终仍失败时
    把臂收回 home，避免举着臂去导航。

    一次调用从头跑到尾（单轮十几到几十秒；够不着重试时每次多约 20 秒）。
    """
    started = time.monotonic()
    ok, stage, message, result = _run("grasp", request.object_name)
    elapsed = time.monotonic() - started
    if not ok:
        # Pick.srv 没有 stage 字段：把阶段名放进 message 开头，调用方/日志都能定位。
        message = f"{stage}: {message}"
    log.info("pick 用了 %.1f s：%s", elapsed, message)
    return Pick_Response(
        success=bool(ok),
        message=message,
        grasp_pose=_grasp_pose_from_result(result),
        gripper_width=_gripper_width_from_result(result),
        score=0.0,
        elapsed_s=elapsed,
    )


@skill.mcp(D_OBSERVE)
def observe(request: Observe_Request) -> Observe_Response:
    """回观察位看一眼并报出结论，臂不做任何抓取动作。

    用于抓之前确认现场、或抓失败后查原因：报出检测框、顶面中心、承托面的朝向、
    闭合方向、投影宽度、量到的物高，以及照这些量算出来的计划过不过关。
    只走"回观察位 + 张开夹爪"这一步，比抓取快得多。
    """
    ok, stage, message, result = _run("observe", request.target)
    if not ok and stage not in ("observe", "inactive", "target", "data", "exception"):
        message = f"{stage}: {message}"
    return Observe_Response(
        success=bool(ok),
        message=message,
        detail_json=json.dumps(result, ensure_ascii=False, default=_json_default),
    )


def _put_down(config: Config | None) -> tuple:
    """放下：可选先到放点 → 张开夹爪到最大 → 回安全位。返回 (ok, message)。"""
    client = _state.client
    pipeline = _state.pipeline
    if client is None or pipeline is None:
        return False, "技能还没激活（第一次调用前 executor 才会激活它）"
    config = config or Config()
    with _state.lock:                              # 一条臂同一时刻只能有一件事
        notes = []
        if config.put_down_joints_deg is not None:
            actual = client.move_joints(config.put_down_joints_deg)
            notes.append("先到放点（实际 %s°）"
                         % [round(float(v), 1) for v in actual])
        else:
            notes.append("没配 put_down_joints_deg，就在当前位放开")
        opening = pipeline.opening_m()
        actual = client.gripper(opening)
        notes.append("夹爪开到 %.0f mm（实际 %.0f mm）"
                     % (opening * 1000.0, float(actual) * 1000.0))
        safe = (config.teach_safe_joints_deg
                if config.teach_safe_joints_deg is not None else [0.0] * 6)
        actual = client.move_joints(safe)
        notes.append("回安全位 %s°（实际 %s°）"
                     % ([round(float(v), 1) for v in safe],
                        [round(float(v), 1) for v in actual]))
    return True, "放下完成：" + "；".join(notes)


@skill.mcp(D_PUTDOWN)
def put_down(request: PutDown_Request) -> PutDown_Response:
    """放下手里握着的东西：张开夹爪并回到安全位。

    用户说"放下 / 松手 / put it down"时调用，请求没有字段。流程：
      1. 配了 `put_down_joints_deg` 就先把臂开到那个放点（没配就在当前位）；
      2. 夹爪张开到最大开口（标定里的满开口，没有标定就给默认值）；
      3. 回 `teach_safe_joints_deg`（没配回全零 home）。

    它**不搬运**：要搬到别处不是这个工具的事。抓取成功后物体还握在爪里，直到调用它。
    """
    started = time.monotonic()
    try:
        ok, message = _put_down(_state.config)
    except Exception as error:                     # noqa: BLE001 —— 工具失败要如实返回
        ok = False
        message = f"{type(error).__name__}: {error}"
    if not ok:
        message = "put_down: " + message
    log.info("put_down 用了 %.1f s：%s", time.monotonic() - started, message)
    return PutDown_Response(success=bool(ok), message=message,
                            elapsed_s=time.monotonic() - started)


def main() -> int:
    skill.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
