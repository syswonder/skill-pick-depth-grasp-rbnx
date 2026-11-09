#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
#
# 本包没有自己拉起的外部进程（ROS 节点与视觉 gRPC 通道由 Driver 生命周期在进程内
# 建/拆，进程退出时回收）。stop 只做幂等清理：把"已构建"标记删掉，下次 rbnx 会重新
# 跑 build。codegen 产物本身留着——删了它也不影响正确性，只是下次启动会多等一次
# codegen，而且停一次包就删产物会让 `rbnx boot` 的后续启动变慢。
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
rm -f -- "$PKG/rbnx-build/.rbnx-built"
exit 0
