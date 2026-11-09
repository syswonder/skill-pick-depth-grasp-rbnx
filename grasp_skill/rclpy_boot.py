# SPDX-License-Identifier: MulanPSL-2.0
"""rclpy 的起停。技能会被反复激活/停用，所以这两件事都得幂等。

技能和普通 ROS 节点不一样：进程常驻，但真正用到 ROS 只在激活之后。所以 init 放在
on_activate、shutdown 放在 on_deactivate，而不是进程一起来就 init。
"""
from __future__ import annotations

import logging

log = logging.getLogger("depth_grasp")


def ensure_init() -> None:
    """rclpy.init()，已经初始化过就什么都不做。"""
    import rclpy

    if not rclpy.ok():
        rclpy.init()


def ensure_shutdown() -> None:
    """rclpy.shutdown()，已经关了也不报错。"""
    try:
        import rclpy

        if rclpy.ok():
            rclpy.shutdown()
    except Exception as error:                     # noqa: BLE001 —— 关到一半再抛没意义
        log.warning("rclpy 关闭时出错（忽略）：%s", error)
