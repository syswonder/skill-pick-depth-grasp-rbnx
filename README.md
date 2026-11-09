# skill-pick-depth-grasp-rbnx

深度几何抓取技能（`robonix/skill/pick/*`）的 `depth_grasp` 实现。能力面（pick /
put_down / observe 的调用条件、参数、返回、失败前缀、config 键表）见
[`CAPABILITY.md`](CAPABILITY.md)。

## 它是什么

一条"看一眼 → 算抓取点 → 下爪 → 合爪 → 抬起 → 保持握持"的真机流水线。与
`skill-pick-vertical-grasp-rbnx`（`vertical_grasp`）是同一个 `robonix/skill/pick`
命名空间下的两个实现，**一个部署只能起一个**：

- `vertical_grasp`：bbox + 2D 单应 + 固定桌面高度；
- 本包：一帧 RGB-D → 承托面/物体顶面/长轴/闭合方向/闭合宽度，**不依赖固定桌面高度**，
  斜面上、物体被挪过、检测框偏了都不影响定位精度。

视觉算法**不在本包**：它在服务 `robonix/service/perception/grasp_geometry/locate`
里（检测、承托面拟合、顶面提取、"框和角度换算成三维"）。服务自己订阅相机、取最新
一帧、输出**相机系**几何量；本技能经 atlas 解析它的 gRPC 端点后调 `Locate`，再用
标定里的手眼外参把结果搬到臂基座系。

因此本包：

- **不订阅任何相机话题**；`client.py` 只有臂的状态/命令；
- **不依赖宿主 HTTP 服务**（老的 `host-services/grasp-vision` 那条路已经删掉）；
- **不碰 VLM 凭据**（提示词、模型、密钥都在服务配置里）；
- 视觉调参/换模型 → 改服务、重启服务（一秒），不用重启 rbnx。

## 包内布局

```
grasp_skill/
  atlas_bridge.py   能力声明 + 生命周期 + 三个工具（pick / put_down / observe）的编排
  vision_service.py 视觉服务的 gRPC 客户端：atlas 解析端点、调 Locate、失败分类
  pipeline.py       八步流程（拍照 → 视觉量一眼 → 计划 → 悬停 → 下爪 → 合爪 → 抬起 → 回位）
  client.py         ROS 侧的臂/夹爪；话题名由 atlas_bridge 解析后注入（**无相机**）
  observe_scan.py   多观察位扫描：标称位看不准/量不出来时规划几个备用观察位（纯 numpy）
  grasp.py motion.py lm_ik.py   纯几何、轨迹与数值 IK（不碰像素）
  calib/            标定产物读取与 IK 链（手眼外参、爪尖、观察位、夹爪尺寸、几何修正）
  real_adapter.py   把部署配置拼成流水线认的 scene
  backoff.py        够不着时的退避重试（回 home → 站 → 前进 → 趴 → 再抓）
  chassis.py        底盘最小客户端（站/趴/前进，走底盘原语的 MCP/gRPC）
  rclpy_boot.py     rclpy 的幂等起停
capabilities/
  pick.v1.toml / put_down.v1.toml / observe.v1.toml  本包的三个 MCP 契约
  lib/pick/srv/{Pick,PutDown,Observe}.srv
  service/perception/grasp_geometry/locate.v1.toml   要消费的服务契约副本
  lib/grasp_geometry/srv/Locate.srv                  （codegen 生成 gRPC 客户端桩）
tests/
  fakes.py                       解析几何假臂（FakeCalibration）
  test_observe_scan.py           observe_scan 纯单测（不依赖 ROS/相机/服务）
  test_pipeline_observe_scan.py  pipeline 多观察位扫描接线测试（假 client + 假 _look）
  test_observe_scan_acceptance.py 独立验收：几何约束重算 + 视觉客户端契约
scripts/
  build.sh   rbnx codegen → rbnx-build/codegen/{proto_gen,robonix_mcp_types}
  start.sh   source ROS humble → PYTHONPATH 加 codegen 产物 → exec 主模块
  stop.sh    幂等清理（删 `.rbnx-built` 标记）
```

## 依赖与部署前提

1. 臂/夹爪原语 `robonix/primitive/arm/{joint_states, joint_command, end_pose}` 在
   atlas 上可解析（技能激活时逐个解析端点；解析不到 → 激活失败，不猜默认话题）。
2. 视觉服务 `robonix/service/perception/grasp_geometry/locate` **ACTIVE**。技能在
   `on_activate` 里等它最多 60 s；等不到 → 激活失败。服务的相机绑定、内参、检测模型/
   VLM 凭据都在**服务**的配置里。
3. 标定：手眼外参 `gripper_camera`、爪尖偏移 `tip_in_flange`、观察位
   `observe_joints_deg`、夹爪尺寸 `gripper_size`、臂几何修正 `correction`（可选）。
   相机内参/畸变系数不再是技能必需项。
4. 底盘（可选）：`primitive/chassis/{move,posture}` 只在够不着退避重试时用。

## 构建 / 起停

```bash
bash scripts/build.sh      # rbnx codegen -p . --mcp --out-dir rbnx-build/codegen
bash scripts/start.sh      # exec python3 -m grasp_skill.atlas_bridge
bash scripts/stop.sh       # 幂等清理
```

技能是**懒激活**的：`rbnx boot` 只发 `CMD_INIT`（只解析配置），停在 INACTIVE；
第一次被调用之前 executor 才发 `CMD_ACTIVATE`，那一刻才解析原语/视觉服务端点、建
标定、起 ROS 节点。所以 `rbnx caps` 里看到它 `[INACTIVE]` 是正常的，不是故障。
`CMD_DEACTIVATE` 会把 ROS 节点与视觉 gRPC 通道拆干净，可反复激活。

```bash
rbnx validate .            # 检查 manifest
```

## 配置

完整键表见 [`CAPABILITY.md`](CAPABILITY.md) 的"配置"一节。最小示例：

```yaml
# 部署清单的 skill 条目
- name: depth_grasp
  path: ./packages/skill-pick-depth-grasp-rbnx
  config:
    calib_source: inline
    calib:
      gripper_camera:        # T_flange_camera，4×4 行优先（标定台导出的数）
        [[1,0,0,0.0],[0,1,0,0.0],[0,0,1,0.09],[0,0,0,1]]
      tip_in_flange: [0.0, 0.0, 0.142]     # 爪尖相对法兰（米）
      observe_joints_deg: [0, 60, -50, 0, 70, 0]
      gripper_size: {clamp_length_mm: 72, opening_mm: 102, closed_gap_mm: 0}
      correction: null                     # 可选，臂几何修正
    table_height_m: -0.05                  # 只用于诊断/扫描兜底
    vision_timeout_s: 180
    observe_scan:
      enabled: true
      max_views: 4
      anchors_deg: []                      # 人示教的额外观察锚点（每组 6 个关节角）
    put_down_joints_deg: null              # 放点（空 = 当前位放开）
    teach_safe_joints_deg: null            # 空 = 回全零 home
```

`calib_source: artifacts` 时给 `calib_root` 指标定产物目录，技能直接读（开发台/标定台
旁用）；机器人容器里没有 `tools/`，用 `inline`。

## 测试

```bash
cd <包目录>
source /opt/ros/humble/setup.bash
PYTHONPATH="rbnx-build/codegen/proto_gen:rbnx-build/codegen/robonix_mcp_types:.:$(rbnx path robonix-api):$PYTHONPATH" \
  python3 -m pytest tests/ -q
```

测试全部不连 ROS / 相机 / atlas / 视觉服务：

- `test_observe_scan.py`：多观察位规划与支撑面估计（纯 numpy + 解析假臂）；
- `test_pipeline_observe_scan.py`：`_look` 被替身替换，测 `_observe` 的扫描行为与
  "哪类失败才值得换观察位"；
- `test_observe_scan_acceptance.py`：独立验收——把规划结果拿回假臂逐条重算几何约束、
  重算支撑面估计、以及 `vision_service` 的失败分类 / `result_json` 解析。

## 已知限制 / TODO / 风险

1. **多观察位扫描的支撑面先验会退到配置常量**。规划观察位需要"支撑面大概多高"；
   服务只在成功时给承托面，而扫描只在"这一眼没量出来"时触发——失败响应不带宽高信息。
   所以真机上这一层目前退到 `table_height_m`。要彻底解决：服务失败时也交回近似的面，
   或加一条"只估面"的轻量调用（契约变更，需两边一起改）。`table_height_m` 请按现场
   实测填写，否则自动扫描可能把观察位规划到错误高度。
2. **`put_down` 的放点来源**：只认配置 `put_down_joints_deg`（或当前位）。没有"把手里
   东西放到某个视觉目标/盒子"的能力——那是 `vertical_grasp` 的 `put_in_box`，本包没有。
   要按物体/台面自动决定放点，得加坐标来源（目前没有）。
3. **`pick` 响应里的 `grasp_pose`** 取的是"下爪停住时臂实到的法兰位姿"（标定链正解），
   不保证等于计划里那个理想位姿（伺服有残差）。没走到下爪就失败时给零位姿。若下游要
   "计划位姿"，需要在 `stages.descend` 里另取（`commanded_joints_deg` / `grasp_tip_m`）。
4. **失败分类按文案**：视觉服务契约只有 `success/message`，没有机器可读的 kind。
   `vision_service.classify_failure()` 按中文文案把失败分成 measurement / server /
   request；服务改了文案分类就会漂（只会影响"要不要换观察位"，不会把失败当成功）。
   建议服务契约后续加一个 `kind` 或 `error_code` 字段。
5. **`observe` 比旧包多做了"量一眼"**：旧实现的 observe 只回拍照位、不真的测量；本版按
   `Observe.srv` 的语义报出测量与计划结论（失败时 `success=false`）。依赖视觉服务的
   部署才具备这个能力。
6. **没有在真机上跑过一轮**：代码路径与旧 `s10_grasp_dev` 一致（只换了视觉入口与响应
   契约），单测/codegen/import 都已通过，但真机到位精度、扫描、退避重试仍需上机验。
7. `out_dir` 默认空 = 不落盘；要复盘时配一个目录，只落结果 JSON（`result.json` /
   `request_state.json` / `detector.json`）。原始帧在视觉服务 `log_dir` 的
   `<request_id>/` 下（`observe` 的 `detail_json.measurement.request_id` 可对上）。

## License

MulanPSL-2.0（见 `LICENSE`）。
