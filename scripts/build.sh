#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
#
# 从本包的 capabilities/*.v1.toml + capabilities/lib/ 下的 srv 生成两类产物：
#   rbnx-build/codegen/proto_gen          gRPC 桩与消息类
#     - robonix_contracts_pb2_grpc：本包三个 MCP 契约 + 要消费的视觉服务契约
#       （RobonixServicePerceptionGraspGeometryLocateStub）
#   rbnx-build/codegen/robonix_mcp_types  MCP 工具签名用的请求/响应 dataclass
#     - pick_mcp（Pick/PutDown/Observe）+ geometry_msgs_mcp 等嵌套类型
# start.sh 要求这两份都在，缺了直接退出——不要拿没生成的桩去起服务。
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"

if [[ "${RBNX_BUILD_CLEAN:-0}" == "1" ]]; then
    rm -rf rbnx-build/codegen rbnx-build/proto-staging
fi
mkdir -p rbnx-build

rbnx codegen -p "$PKG" --mcp --out-dir "$PKG/rbnx-build/codegen"

touch rbnx-build/.rbnx-built
echo "[depth_grasp/build] codegen 完成"
