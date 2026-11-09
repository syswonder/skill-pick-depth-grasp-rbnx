# Description: 抓取技能用的最小底盘客户端——站/趴/前进，走底盘原语的 MCP（gRPC）。
#
# 为什么走 MCP 而不是直接发 /MOTION_STATE、/NAV_CMD：
#   - 站立/趴下是原语私有契约（drdds MotionState），技能里不该复制那套协议；
#   - chassis/move 自带"必须在 RL 导航模式"的检查和速度上限，直接发 /NAV_CMD
#     会绕过这些保护。
#
# 端点由 atlas 解析（和 observe 技能同一套做法）：
#   ATLAS.find_unique_capability(contract_id, transport="grpc")
#   skill.connect_capability(view, contract_id, "grpc") -> channel.endpoint
#
# 服务名有历史包袱：当前原语注册的是 RobonixPrimitiveChassisPosture，旧版 codegen
# 里叫 RobonixPrimitiveQuadrupedPosture。posture 调用两个都试（UNIMPLEMENTED 时换
# 下一个），这样技能不依赖自己那份 codegen 是否刚重新生成过。
from __future__ import annotations

import json
import logging
import time

import grpc

from robonix_api import ATLAS

log = logging.getLogger("depth_grasp.chassis")

MOVE_CONTRACT = "robonix/primitive/chassis/move"
POSTURE_CONTRACT = "robonix/primitive/chassis/posture"

_MOVE_SERVICE = "RobonixPrimitiveChassisMove"
_POSTURE_SERVICES = ("RobonixPrimitiveChassisPosture",
                     "RobonixPrimitiveQuadrupedPosture")


def _clean_endpoint(endpoint: str) -> str:
    """Normalize an atlas gRPC endpoint for grpcio."""
    return (str(endpoint)
            .removeprefix("http://").removeprefix("https://")
            .replace("localhost", "127.0.0.1"))


def resolve(skill, contract_id: str, timeout_s: float = 10.0) -> str | None:
    """Resolve one capability's gRPC endpoint; None when unavailable."""
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            view = ATLAS.find_unique_capability(
                contract_id=contract_id, transport="grpc")
            channel = skill.connect_capability(view, contract_id, "grpc")
            endpoint = channel.endpoint
            try:
                channel.close()
            except Exception:                      # noqa: BLE001
                pass
            if endpoint:
                return _clean_endpoint(endpoint)
            last_error = "atlas 给的端点为空"
        except Exception as error:                 # noqa: BLE001 —— 还没注册上就继续等
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(1.0)
    log.warning("解析 %s 失败：%s", contract_id, last_error)
    return None


class ChassisClient:
    """站/趴/前进。方法返回 (ok, message)，不抛异常给调用方。"""

    def __init__(self, move_endpoint: str, posture_endpoint: str):
        self._move_endpoint = _clean_endpoint(move_endpoint)
        self._posture_endpoint = _clean_endpoint(posture_endpoint)

    def _call(self, endpoint: str, path: str, request,
              response_type, timeout_s: float):
        channel = grpc.insecure_channel(endpoint)
        try:
            grpc.channel_ready_future(channel).result(timeout=5.0)
            call = channel.unary_unary(
                path,
                request_serializer=type(request).SerializeToString,
                response_deserializer=response_type.FromString,
            )
            return call(request, timeout=timeout_s)
        finally:
            channel.close()

    # ---------- posture ----------

    def set_posture(self, name: str, timeout_s: float = 30.0) -> tuple[bool, str]:
        import quadruped_pb2

        request = quadruped_pb2.SetPosture_Request(posture_name=str(name))
        last_error = None
        for service in _POSTURE_SERVICES:
            try:
                response = self._call(
                    self._posture_endpoint,
                    f"/robonix.contracts.{service}/SetPosture",
                    request, quadruped_pb2.SetPosture_Response, timeout_s)
            except grpc.RpcError as error:
                last_error = error
                if error.code() == grpc.StatusCode.UNIMPLEMENTED:
                    continue
                return False, f"{service}: {error.code().name}: {error.details()}"
            return bool(response.success), str(response.message)
        return False, f"没有可用的 posture 服务（{last_error}）"

    # ---------- move ----------

    def move_forward(self, meters: float,
                     timeout_s: float = 20.0) -> tuple[bool, str]:
        import chassis_pb2

        request = chassis_pb2.ExecuteMoveCommand_Request(
            command=chassis_pb2.MoveCommand(forward_m=float(meters)))
        try:
            response = self._call(
                self._move_endpoint,
                f"/robonix.contracts.{_MOVE_SERVICE}/ExecuteMoveCommand",
                request, chassis_pb2.ExecuteMoveCommand_Response, timeout_s)
        except grpc.RpcError as error:
            return False, f"{error.code().name}: {error.details()}"
        raw = response.status.data or "{}"
        try:
            status = json.loads(raw)
        except (TypeError, ValueError):
            return False, f"返回的 status 不是 JSON：{raw!r}"
        if status.get("status") == "done":
            return True, "done"
        return False, str(status.get("error") or status)


def open_chassis(skill, *, timeout_s: float = 10.0) -> ChassisClient | None:
    """Resolve both endpoints; None when the deployment has no chassis."""
    move = resolve(skill, MOVE_CONTRACT, timeout_s=timeout_s)
    posture = resolve(skill, POSTURE_CONTRACT, timeout_s=timeout_s)
    if not move or not posture:
        log.warning("底盘能力不可用（move=%s, posture=%s）；退避重试关闭",
                    move, posture)
        return None
    log.info("底盘端点：move=%s posture=%s", move, posture)
    return ChassisClient(move, posture)
