# SPDX-License-Identifier: MulanPSL-2.0
"""真机那半**只剩一块**：把部署配置拼成流水线认的 scene。

抄自标定台的「抓取测试」——那边真机复用同一条流水线，这块是它为了让流水线在真机上跑通
而做的适配，做法原样搬来，只把参数来源从页面表单换成本技能的配置文件。

**另外两块都不在这儿了**：
  - "把物体在哪估出来"（原来的 ObjectEstimatingDetector）——见文件末尾的说明，删了；
    物体在哪现在由视觉服务算（robonix/service/perception/grasp_geometry/locate）。
  - client：真机的臂/相机在技能里是原语（ROS 话题，名字经 atlas 解析），不是标定台
    那套硬件，所以用本包那份 client.py——它只多补了 holding()。
"""
from __future__ import annotations

import numpy as np


def scene_from_config(config) -> dict:
    """按标定台那份配方拼 scene：位置那半由"看"那一眼现测，这里只给常量。

    - 物体尺寸是**人量的**（配置里 length_mm / width_mm / height_mm），不是猜的；
      序号固定 0（真机上只有一个目标）。
    - 承托面给常量：桌面是平的，法线 [0,0,1]，点的高度取量出来的桌面高度。
      它**只在流水线的诊断里**跟拟合出来的平面对一对（normal_error_deg），
      不参与任何运动决策——真正用的是那一眼从深度图里拟合出来的那个面。
    """
    half = [float(config.object_size_mm[key]) / 1000.0 / 2.0
            for key in ("length_mm", "width_mm", "height_mm")]
    return {
        # 名字留空：真正的名字由每次调用（MCP 的 target）填进来，见 atlas_bridge._run。
        # 这里不放默认值——放了就又要回答"配置和调用方谁说了算"。
        "target": {"index": 0, "name": "", "half_size": half},
        "ramp": {"plane": {"point": [0.0, 0.0, float(config.table_height_m)],
                           "normal": [0.0, 0.0, 1.0]}},
        "environment": "real",
    }


# ObjectEstimatingDetector 删了。它原来干两件事：
#   1. 每认一次就往 client 的物体表里补一条"物体在哪"——那是**从检测框中心反投影**估的，
#      实测会跟着框跑（框外扩 1.4× 就偏 31 mm）；物体在哪现在由**视觉服务**算（从深度图
#      分离物体、取顶面质心，框偏了它不跟着跑）。
#   2. 自己又去取一帧深度（当年 client 还有相机订阅时"要一帧比上次新的"），拿到的
#      **不是**流水线手里那帧——一个适配器跑出去取帧，本身就是没切开的表现。
#      （现在帧由视觉服务自己取，技能端连相机订阅都没有了，这一条更无从谈起。）
#
# 它喂的那张物体表在真机上本来就是空的（没有 /sim/objects），所以删掉之后：真机上凡是要
# 读"物体位姿"的地方都返回 None，相关判据退回 client.holding()（见 pipeline 的
# object_position / object_at_look）。
