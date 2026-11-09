# SPDX-License-Identifier: MulanPSL-2.0
"""视觉服务的 gRPC 客户端：消费 `robonix/service/perception/grasp_geometry/locate`。

视觉算法（检测、承托面拟合、顶面提取、"框和角度换算成三维"）已经搬到 service 包里，
**不在本技能里**。技能这边只做三件事：

    1. 经 atlas 找到这个服务的 gRPC 端点（`find_unique_capability` + connect）；
    2. 把"找什么"发过去（`Locate_Request{target, box}`）；
    3. 把 `result_json` 解析成 dict 交回 pipeline。

结果一律是**相机系**（x 右、y 下、z 前，OpenCV 约定）。要落到臂/底盘基座系，
由调用方乘一次手眼外参（pipeline 的 `camera_pose`，来自标定 `gripper_camera`）。

**失败语义**（`VisionError.kind`）：调用方（`pipeline._observe`）据此判断
"换个观察位再看"有没有意义——只有 `measurement` 值得换，其它换多少角度都一样：

    transport     连不上服务 / gRPC 超时 / 通道坏
    server        服务在、但这一次请求没被受理（没初始化、上一次还没处理完、
                  VLM 凭据不全…）或返回体坏掉
    request       请求本身不合法（target/box 都不给、box 格式错）
    measurement   服务受理了，但这一眼没量出来（目标不在画面、深度无效、
                  承托面拟合失败…）——**只有这一类可以换观察位重试**

契约见包内 `capabilities/service/perception/grasp_geometry/locate.v1.toml` 与
`capabilities/lib/grasp_geometry/srv/Locate.srv`。codegen 生成的是
`robonix_contracts_pb2_grpc.RobonixServicePerceptionGraspGeometryLocateStub`
和 `grasp_geometry_pb2.Locate_Request / Locate_Response`。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

log = logging.getLogger("depth_grasp.vision")

CONTRACT_ID = "robonix/service/perception/grasp_geometry/locate"
# 服务自己的单次 locate 预算默认 120 s（含 VLM 检测）；客户端留出余量。
DEFAULT_TIMEOUT_S = 180.0
RESOLVE_TIMEOUT_S = 60.0

# 成功返回体里必须有的键：少了就是服务侧契约漂了，不能拿半份结果往下走。
REQUIRED_RESULT_KEYS = ("top_point", "support", "long_axis", "closing", "bbox")


class VisionError(RuntimeError):
    """一次视觉调用失败。

    Args:
        message: 给人看的一句话（测量失败会带 "这一眼没量出来：" 前缀）。
        kind: transport / server / request / measurement；未知时 None（旧行为）。
    """

    def __init__(self, message: str, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


def classify_failure(message: str) -> str:
    """把服务返回的失败文案归到 kind 里。

    契约目前只给 `success/message`，没有机器可读的 kind，所以按文案分类。
    分类只影响"要不要换观察位"这一件事，分错最坏的后果是多走或少走几个观察位，
    不会把失败硬掰成成功。服务哪天加了 kind 字段，这里应该优先用它。
    """
    text = (message or "").strip()
    if text.startswith("请求非法"):
        return "request"
    for marker in ("还没初始化", "还没处理完", "凭据不全", "没有收到 CMD_INIT"):
        if marker in text:
            return "server"
    return "measurement"


def _clean_endpoint(endpoint: str) -> str:
    """atlas 上的 gRPC 端点有时带 scheme/localhost，统一成 grpcio 认的 host:port。"""
    return (str(endpoint or "")
            .removeprefix("http://").removeprefix("https://")
            .replace("localhost", "127.0.0.1"))


def resolve_endpoint(skill, *, provider_id: str = "",
                     timeout_s: float = RESOLVE_TIMEOUT_S) -> str:
    """等 atlas 上出现这个服务并返回它的 gRPC 端点。解析不到就抛，不返回兜底值。

    Args:
        skill: 本技能的 `Skill` 实例（用于 `connect_capability` 记账）。
        provider_id: 可选，多实例部署时用来去歧义（config.vision_provider_id）。
        timeout_s: 最多等多久（服务可能比技能晚起）。

    Raises:
        RuntimeError: 截止时间到了还没解析到可用端点。
    """
    from robonix_api import ATLAS

    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
    while time.monotonic() < deadline:
        try:
            kwargs: dict = {"contract_id": CONTRACT_ID, "transport": "grpc"}
            if provider_id:
                kwargs["provider_id"] = provider_id
            view = ATLAS.find_unique_capability(**kwargs)
            channel = skill.connect_capability(view, CONTRACT_ID, "grpc")
            endpoint = _clean_endpoint(channel.endpoint)
            try:
                channel.close()
            except Exception:                      # noqa: BLE001 —— 关不掉不影响解析
                pass
            if endpoint:
                return endpoint
            last_error = "atlas 给的端点为空"
        except Exception as error:                 # noqa: BLE001 —— 还没注册上就继续等
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(1.0)
    raise RuntimeError(
        f"atlas 上解析不到视觉服务 {CONTRACT_ID}"
        + (f"（provider_id={provider_id!r}）" if provider_id else "")
        + f"：{last_error}。这个技能必须有 service/perception/grasp_geometry 在跑。")


class VisionClient:
    """`locate` 的 gRPC 客户端。只负责发请求、把响应变成 dict。"""

    def __init__(self, endpoint: str, *, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.endpoint = _clean_endpoint(endpoint)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._channel = None
        self._stub = None

    # ---------- 连接 ----------

    def _stub_or_raise(self):
        """惰性建通道与桩（activate 时只解析端点，第一次调用才连）。"""
        with self._lock:
            if self._stub is not None:
                return self._stub
            import grpc
            import robonix_contracts_pb2_grpc as contracts_grpc

            channel = grpc.insecure_channel(
                self.endpoint, options=[("grpc.enable_http_proxy", 0)])
            self._channel = channel
            self._stub = (
                contracts_grpc.RobonixServicePerceptionGraspGeometryLocateStub(channel))
            return self._stub

    def close(self) -> None:
        with self._lock:
            if self._channel is not None:
                try:
                    self._channel.close()
                except Exception:                  # noqa: BLE001
                    pass
            self._channel = None
            self._stub = None

    # ---------- 一次调用 ----------

    def locate(self, target: str = "", box=None) -> dict:
        """量一眼，返回相机系的几何量 dict。

        Args:
            target: 开放词表目标名；服务用它做 VLM 检测。
            box: 可选检测框 `[x0, y0, x1, y1(, angle_deg)]`；给了就跳过检测。

        Returns:
            服务 result_json 解析出的 dict（至少含 top_point / support /
            long_axis / closing / bbox / diagnostics）。

        Raises:
            VisionError: 见模块头的 kind 说明。
        """
        import grpc
        import grasp_geometry_pb2 as geometry_pb

        box_text = ""
        if box is not None:
            try:
                values = [float(v) for v in box]
            except (TypeError, ValueError) as error:
                raise VisionError(f"box 不是数：{error}", kind="request") from error
            box_text = json.dumps(values)

        request = geometry_pb.Locate_Request(
            target=str(target or "").strip(), box=box_text)
        try:
            stub = self._stub_or_raise()
            response = stub.Locate(request, timeout=self.timeout_s)
        except VisionError:
            raise
        except grpc.RpcError as error:
            code = ""
            details = ""
            try:
                code = str(error.code().name)
                details = str(error.details() or "")
            except Exception:                      # noqa: BLE001
                details = str(error)
            raise VisionError(
                f"调用视觉服务失败（{self.endpoint}）：{code or type(error).__name__}"
                f" {details}".strip(), kind="transport") from error
        except Exception as error:                 # noqa: BLE001 —— 桩/序列化异常
            raise VisionError(
                f"调用视觉服务失败（{self.endpoint}）：{type(error).__name__}: {error}",
                kind="transport") from error

        if not response.success:
            message = str(response.message or "").strip() or "服务没给原因"
            request_id = ""
            try:
                failure = json.loads(response.result_json or "{}")
                request_id = str(failure.get("request_id") or "")
            except (TypeError, ValueError):
                pass
            detail = (f"（服务那次的现场：{request_id}）" if request_id else "")
            raise VisionError(f"这一眼没量出来：{message}{detail}",
                              kind=classify_failure(message))

        try:
            payload = json.loads(response.result_json or "{}")
        except (TypeError, ValueError) as error:
            raise VisionError(
                f"视觉服务返回的 result_json 不是 JSON：{error}", kind="server") from error
        if not isinstance(payload, dict):
            raise VisionError(
                f"视觉服务返回的 result_json 不是对象（{type(payload).__name__}）",
                kind="server")
        if payload.get("ok") is False:
            why = str(payload.get("why") or response.message or "这一眼没量出来")
            raise VisionError(f"这一眼没量出来：{why}", kind="measurement")
        missing = [key for key in REQUIRED_RESULT_KEYS if key not in payload]
        if missing:
            raise VisionError(
                f"视觉服务返回的结果缺字段 {missing}（服务契约漂了？）", kind="server")
        return payload


def open_client(skill, config) -> VisionClient:
    """按配置解析端点并建客户端（给 on_activate 用）。"""
    endpoint = resolve_endpoint(
        skill, provider_id=getattr(config, "vision_provider_id", "") or "")
    client = VisionClient(
        endpoint, timeout_s=getattr(config, "vision_timeout_s", DEFAULT_TIMEOUT_S))
    log.info("视觉服务 %s → %s（超时 %.0f s）", CONTRACT_ID, endpoint, client.timeout_s)
    return client
