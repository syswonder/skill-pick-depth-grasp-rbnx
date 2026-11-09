---
description: 深度几何抓取技能。pick 抓一次并保持握持（成功时回 home 位、夹爪闭合，不释放不搬运）；put_down 松开夹爪并回安全位；observe 回观察位看一眼、把测量与计划结论报出来（只量不动）。物体在哪、顶面多高由 service/perception/grasp_geometry 用一帧 RGB-D 现量，不依赖固定桌面高度和 2D 单应；技能端不订阅相机。目标不在画面、承托面量不出来、下爪没到位等情况会如实返回卡在哪一步以及原因，不会硬合爪。
---

# pick.depth_grasp（深度几何抓取技能）

命名空间 `robonix/skill/pick`，本包注册三个可调用能力：

| 契约 id | 干什么 | 什么时候调 |
|---|---|---|
| `robonix/skill/pick/pick` | 抓起来并**保持握持**（成功时回 home 位、夹爪闭合） | 用户说"抓起来 / 把它拿起来 / grab it" |
| `robonix/skill/pick/put_down` | **松开**夹爪、回安全位（不搬运） | 用户说"放下 / 松手 / put it down" |
| `robonix/skill/pick/observe` | 回观察位**看一眼并报出结论**（只量不动） | 用户说"看看现在能不能抓 / 看它在哪 / 检查一下" |

`driver` 由框架自动注册共享的 `robonix/lifecycle/driver`，不在本包能力面里重复声明。

## 与 vertical_grasp 的区别（同一个命名空间的两个实现）

`robonix/skill/pick/*` 下有两个实现：本包（`depth_grasp`）与
`skill-pick-vertical-grasp-rbnx`（`vertical_grasp`），两者实现同一组契约 id。**一个部署里只能起一个 pick 实现**，
否则同一个契约 id 会有两个 provider，atlas 解析时会报歧义。

两者对外语义一样（抓起来并保持，不释放）；差别在"物体在哪、顶面多高"这一段怎么来：

- `vertical_grasp`：bbox + 2D 单应（homography）+ **固定桌面高度** → 一个垂直抓取位姿。
  物体被挪过、摆在斜面上、桌面高度变了，就会偏。
- `depth_grasp`（本包）：服务用一帧 RGB-D 把**承托面和物体顶面分开**，量出顶面中心、
  承托面朝向、长轴/闭合方向、闭合方向上的投影宽度，再由技能算抓取点。**不依赖固定
  台面高度**，斜面上、检测框偏了都不影响定位精度。

## 依赖（部署前必须就绪）

1. **臂/夹爪原语**：`robonix/primitive/arm/{joint_states, joint_command, end_pose}`。
   话题名由技能在激活时经 atlas 解析，不写死；解析不到就激活失败（不猜默认话题）。
2. **视觉服务**：`robonix/service/perception/grasp_geometry/locate` 必须 **ACTIVE**。
   服务自己订阅相机（`primitive/camera/{rgb,depth}`，绑定它的配置），技能只调它的
   `Locate(target, box)` 拿**相机系**结果，再乘技能标定里的手眼外参 `gripper_camera`
   搬到臂基座系。服务没起来时技能激活失败——**技能端不订阅相机、不依赖宿主 HTTP
   服务、不碰 VLM 凭据**。
3. **底盘（可选）**：`primitive/chassis/{move,posture}` 只在"够不着"退避重试时用；
   没有底盘的部署照样能抓，只是不重试。
4. **标定**：手眼外参 `gripper_camera`、爪尖偏移 `tip_in_flange`、观察位
   `observe_joints_deg`、夹爪尺寸 `gripper_size`、臂几何修正 `correction`。
   相机内参/畸变系数/检测模型/VLM 凭据都是**服务**的配置，不在这里传。

## pick — 抓一次并保持握持

请求（`Pick.srv`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `object_name` | string | 是 | 开放词表目标名（中文/英文都行，越具体越好，如 `"红色方块"`、`"tissue pack"`）。服务用它做 VLM 检测。 |

返回（`Pick.srv`）：

| 字段 | 说明 |
|---|---|
| `success` | 真抓着并保持住了（不是"动作发完了"） |
| `message` | 一句话。**失败时以阶段名开头**，见下文"失败怎么读" |
| `grasp_pose` | 本轮**实际执行**的抓取位姿（下爪停住时臂实到的法兰位姿，frame=`arm/base_link`）。没走到下爪就失败时是零位姿（位置全零、单位四元数），**没有编造** |
| `gripper_width` | 合爪停住时实测开口（米）；没合爪就是 0 |
| `score` | 本实现固定 0（没有启发式打分） |
| `elapsed_s` | 整次调用耗时（秒，实测） |

流程：回拍照位 → 调视觉服务量一眼（看不准且是"没量出来"时换几个观察位再看，见下）→
算抓取点 → 悬停 → 下爪 → 合爪加压 → 抬起 → 回 home 位。**成功时停在 home 位、夹爪
闭合、手里握着**，不释放也不搬运；要放下是 `put_down`。

一次调用从头跑到尾（单轮十几到几十秒；够不着退避重试时每次多约 20 秒），期间臂独占。
**默认底盘保持趴下/坐姿，不要为了抓取让底盘站立**；只有确实需要移动底盘时才站，移动
完再趴下。

## put_down — 放下并回安全位

请求（`PutDown.srv`）：无字段。

返回：`success` / `message` / `elapsed_s`。

实现顺序（`config` 见下表）：

1. 配了 `put_down_joints_deg` 就先把六轴开到那个放点；**没配就在当前位**，不动臂；
2. 夹爪张开到最大开口（标定里的满开口；没有夹爪标定就用默认值）；
3. 回 `teach_safe_joints_deg`（**没配回全零 home**）。

它**不搬运**：要搬到别处不是这个工具的事。`failure` 时 `message` 以 `put_down:` 开头。

## observe — 回观察位看一眼，只量不动

请求（`Observe.srv`）：`target`（string，必填，同 pick 的 `object_name`）。

返回：`success` / `message` / `detail_json`。`detail_json` 里是这一眼的：

- `measurement`：顶面中心（基座系，米）、承托面 `{point, normal}`、长轴、闭合方向、
  闭合方向上的投影宽度、服务请求 id（可去服务日志目录看现场帧）；
- `stages.look_photo`：服务返回的诊断（检测来源、每步耗时等）；
- `stages.plan`：照这一眼算出来的计划（物高、爪尖高度、悬停点/抓取点、是否通过及原因）。

臂只走"回观察位 + 张开夹爪"这一步，**不做任何抓取动作**。抓之前确认现场、或抓失败后
查原因用，比 pick 快得多。失败时 `message` 写清"没看准"的原因。

## 失败怎么读（pick 的 message 阶段前缀）

`pick` 失败时 `message` 形如 `descend: 下爪没到位（爪尖差 21.5 mm…）`，前缀是最后走到的
阶段，常见的有：

| 前缀 | 含义 |
|---|---|
| `inactive` | 技能还没被 executor 激活（第一次调用前才会激活） |
| `target` | 没给 `object_name` |
| `data` | 臂的状态话题断了（重建订阅也没回来） |
| `photo` / `look_photo` / `observe_scan` | 回拍照位/看这一眼/换观察位扫描——**看不准或量不出来**，臂不会继续抓 |
| `plan` | 计划没通过（物体太高/太矮、比开口宽、顶面落在承托面之下…） |
| `hover` | 走到悬停位失败（够不着、姿态解不出来） |
| `descend` | 下爪没到计划位置：**不会合爪**（顶着挤会把物体推走），这一轮作废 |
| `close` | 合爪时发现没夹到物体（开口比物体宽出一大截，指头骑在边上/被挡住） |
| `lift` / `return_home` | 抬起来或回位途中出了问题（例如方块滑脱） |
| `exception` | 一轮里抛了异常（类型+原因） |

**如实报**：失败不会拿旧值/默认值兜底，也不会硬凑一个"看起来成功"的结果。把 `message`
原样转述给用户即可；要更细的数字看 `detail_json`（`stages` 每一步的误差、检测框、
承托面、计划判据）。

## 多观察位扫描（看不准时自动换角度）

标称拍照位是标定时教好的**一个固定姿势**，物体偏出画面（或正好被夹爪手指挡住）时这一眼
就白量。补救不是把相机往下压（会撞），而是**换几个观察位各看一眼**：

- 只有"这一眼没量出来"那类失败（`kind=measurement`）才触发；连不上服务/服务报错/传参
  错误换多少角度都一样，立即停。
- 先在标称位周围撒点挑几个合格且彼此差异大的观察位；还不够就依次走到清单
  `observe_scan.anchors_deg` 里人示教/指定的额外锚点再看、再围它扫。看到即停，全看不准
  就回标称位并如实报。
- **已知限制**：规划观察位需要一个"支撑面高度"先验。服务只在成功时给承托面，失败那一
  眼不给；所以真机上这一步目前会退到配置常量 `table_height_m`（见 README 的 TODO）。
  标称位那一眼看着准的时候这个先验用不上（不扫描）。
- 整块不配 = 默认开启、最多 4 个观察位；`observe_scan.enabled: false` 关掉（退回"只在
  标称位看一眼"）。键表见 README。

## 配置（部署清单 `config:` 块）

| 键 | 默认 | 说明 |
|---|---|---|
| `calib_source` | `inline` | `inline`：标定数直接写在下表 `calib` 里（机器人容器用）；`artifacts`：读标定产物目录（开发台/标定台旁） |
| `calib_root` | 空 | `calib_source=artifacts` 时的产物根目录（必填） |
| `calib` | 空 | `calib_source=inline` 时的标定数：`gripper_camera`(4×4)、`tip_in_flange`(3)、`observe_joints_deg`(6)、`gripper_size`{clamp_length_mm,opening_mm,closed_gap_mm}、`correction`（可选）；`camera_matrix`/`dist_coeffs` 已不由技能消费，旧配置里有也兼容 |
| `vision_provider_id` | 空 | 视觉服务多实例时指定 provider id；空 = 要求恰好一个 |
| `vision_timeout_s` | `180` | 调视觉服务 `Locate` 的超时（秒）；要 ≥ 服务自己的 `locate_timeout_s` |
| `out_dir` | 空 | 只落本轮结果 JSON（`result.json` / `request_state.json` / `detector.json`）；**不落原始帧**（帧在服务那边）；空 = 不落 |
| `object_size_mm` | 空 | 物体三围 `{length_mm,width_mm,height_mm}`，**人量的**；用于"握着时开口该多少" |
| `table_height_m` | `0` | 桌面在基座系的高度；只在诊断和"扫描估不出支撑面"兜底时用，**不参与抓取几何** |
| `observe_scan` | 空（默认开启） | 多观察位扫描覆盖项：`enabled` / `max_views` / `anchors_deg` / `spread_levels_deg` / `max_deviation_deg` / `distance_m` / `view_angle_deg` / `aim_radius_m` / `min_rotation_gap_deg` / `min_aim_gap_m` / `aim_forward_min_m` / `seed` |
| `backoff` | 空（默认开启） | 够不着时退避重试：`enabled` / `max_attempts` / `forward_m` / `settle_s` / `home_on_failure` |
| `put_down_joints_deg` | 空 | `put_down` 的放点，6 个关节角（度）；空 = 就在当前位放开 |
| `teach_safe_joints_deg` | 空 | `put_down` 之后的安全位，6 个关节角（度）；空 = 回全零 home |

> 相机内参、畸变系数、检测模型、VLM 凭据都是 **`grasp_geometry` 服务**的配置
> （它自己订阅相机、自己检测），不要写进本技能。视觉算法要改就改服务、重启服务。
