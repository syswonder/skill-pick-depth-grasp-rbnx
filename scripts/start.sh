#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
#
# 起技能进程。source ROS humble 基础环境（rclpy 与 robonix_api 的依赖都要它），
# 把 codegen 的两份产物放进 PYTHONPATH，然后 exec 主模块。
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"

ROS_DISTRO="${ROS_DISTRO:-humble}"
# shellcheck disable=SC1091
set +u; source "/opt/ros/${ROS_DISTRO}/setup.bash"; set -u

CODEGEN_PROTO="$PKG/rbnx-build/codegen/proto_gen"
CODEGEN_MCP="$PKG/rbnx-build/codegen/robonix_mcp_types"
if [[ ! -d "$CODEGEN_PROTO" || ! -d "$CODEGEN_MCP" ]]; then
    echo "[depth_grasp/start] ERR: codegen 产物不在——先跑 scripts/build.sh" >&2
    exit 2
fi
# 末尾必须接 ${PYTHONPATH}：setup.bash 把 rclpy 那一堆路径放在里面，覆盖掉就是
# ModuleNotFoundError: No module named 'rclpy'。
export PYTHONPATH="$CODEGEN_PROTO:$CODEGEN_MCP:$PKG:${PYTHONPATH:-}"
if ROBONIX_API="$(rbnx path robonix-api 2>/dev/null)"; then
    export PYTHONPATH="$ROBONIX_API:$PYTHONPATH"
fi

exec python3 -u -m grasp_skill.atlas_bridge
