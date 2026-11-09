# Description: 多观察位姿规划——标称拍照位看不准时，换几个角度补看一眼。
#
# 为什么要它：真机 2026-10-28 15:19 那一轮报「目标不在腕上相机的画面里」。标称拍照位是
# 标定时**定死的一个姿态**，物体只要偏出那一帧画面（或者干脆被夹爪手指挡住），这一眼就白拍。
# 现场能做的补救不是把相机再往下压（会撞），而是**换几个不同的观察位各看一眼**，看到就停。
#
# 思路照搬标定工具 tools/calibration/eyeinhand/steps/handeye_auto.py 的渐进式手眼标定：
# 在标称位姿附近按档撒关节角 → 用正运动学 + 相机模型把不合格的筛掉 → 挑彼此差异大的。
# 区别只有一处：标定那边板子位置**已知**（board_in_base），能逐点算"板子还在不在画面里"；
# 这里物体在哪**不知道**（"看不准"本身就是没量出来的下场），所以只能用一个"支撑面高度"
# 先验去筛：光轴朝下、落点别出画面、离面距离/水平位置臂够得到、相机和爪尖别扎进桌面。
#
# 这个"支撑面高度"由调用方从刚拍到的那帧深度里估（estimate_surface_z），**不用**配置里的
# table_height_m：那个数在这个现场已经过时了（配置写 -0.0657，深度实测地面在 z≈-0.45 m）。
#
# 纯 numpy：不 import ROS、不碰相机、不打视觉服务。规划只是"算"，走不走、看几眼由 pipeline 定。
import copy

import numpy as np

# 默认参数。跟着**真机这一套几何**定的，不是拍脑袋：标称位相机在基座系
# (0.390, 0.013, 0.432)、光轴 (0.223, 0.054, -0.973)、离地 ~0.95 m。
# 改之前先想清楚哪条约束在拦人——真机跑一遍 notes 里有每级拒绝计数。
SCAN_DEFAULTS = {
    # 总开关：pipeline 只看 plan_views 的返回，这里 false 时它不会规划（也就不会动臂）。
    "enabled": True,
    # 最多走几个备用观察位。4 个是"够看出差别又不至于一轮跑太久"的经验值：
    # 每多一个位姿就多一次"走位 + 拍 + 送视觉服务"，一个位姿真机上 ~5 s。
    "max_views": 4,
    # 渐进撒点：先从小的动量试（稳，离标称近），再放大到更大的角度。两档都撒完再一起挑，
    # 而不是"第一档不够才用第二档"——一起挑才能挑出彼此差异最大的那几个。
    # 单位：度/关节。
    "spread_levels_deg": [[12.0, 8.0, 8.0, 18.0, 15.0, 25.0],
                          [30.0, 18.0, 18.0, 35.0, 30.0, 45.0]],
    # 每档撒多少个样本。真机一次正运动学 ~0.45 ms，两档 600 个 ≈ 0.3 s，可以接受；
    # 再少（比如 100）在窄约束下会挑不满 max_views。
    "samples_per_level": 300,
    # 相对标称位姿**单关节最大偏移**的上限。超过它就别走：臂会大幅换构型，
    # 离开这一片"看得见工作区"的位姿族，走位本身就有风险（撞、绕线、够不到）。
    "max_deviation_deg": 45.0,
    # 光轴与支撑面交点离相机的距离（沿光轴，米）。近了视野里只有一小块、远了分辨率不够；
    # 真机标称位是 0.913 m，所以这区间把标称和"稍微拉远/拉近"的都放进来。
    "distance_m": [0.30, 1.60],
    # 视角：0° = 正俯视，越大越斜。太斜的话深度图误差会被放大，而且斜着看桌面上的物体
    # 容易被自身遮挡；70° 是"还认得出形状"的上限（标定那边 MAX_VIEW_ANGLE 取 50°，
    # 但那是看标定板的精度要求，这里只看"物在不在画面里"，可以松一点）。
    "view_angle_deg": [0.0, 70.0],
    # 光轴落点到基座竖轴的水平距离（米）。下限别在基座正下方（那里臂自己挡着、相机也
    # 对不到）；上限 0.62 是臂的实际可达边界附近——真机标称位的落点半径就有 0.597 m，
    # 再往外伸臂够不着，规划出来也走不过去。
    "aim_radius_m": [0.08, 0.62],
    # 相机离支撑面的最小高度（米）。太低时近处全在盲区（D435i 的最近深度 ~0.28 m），
    # 而且下探本身容易撞到桌面/物体。
    "min_camera_height_m": 0.25,
    # 爪尖离支撑面的最小高度（米）。这条是**安全**约束：备用观察位必须保证爪子还在
    # 物体上方、没有插进桌面；0.08 大致是"两指张开后指板最下沿"的余量。
    "min_tip_height_m": 0.08,
    # 新选的位姿与"标称 + 已选"的**相机旋转**最小夹角（度）。只靠 aim 距离不够：
    # 两个位姿可能对着同一个落点但绕光轴转了 90°，画面朝向完全不同，这才是"换了一角度"。
    "min_rotation_gap_deg": 12.0,
    # 新选的位姿与"标称 + 已选"的**落点**最小水平距离（米）。比这更近的位姿看到的基本
    # 还是同一块地方，走过去纯浪费，所以宁可不扫（< max_views 也停）。
    "min_aim_gap_m": 0.10,
    # 撒点种子。固定住，规划才是确定性的——同一帧失败重试两次不该走两组不同的位姿。
    "seed": 0,
    # **额外观察锚点**：每组 6 个关节角（度）。pipeline 先走标称位看一眼；标称位周围
    # 扫完还不准，就依次走到这些锚点各看一眼，再在它们周围各扫一圈（换一个"站姿"再看，
    # 覆盖/遮挡都不一样）。**空列表 = 只用标称位**，行为与加多锚点之前完全一样。
    # 锚点是人示教/指定的"合格观察位"，规划器不夹它的限位（标称位自己也会超 URDF 限位
    # 一点）；给错了它就走到那个位姿，周围大概率也筛不出合格位姿，不会把抓取带偏。
    "anchors_deg": [],
    # 光轴落点在基座系的 **x 下限**（米）。基座系 +x 是臂的正前方：要抓的物体只会在前方，
    # 往后方扫既没意义（够不着/被机身挡住）又可能把臂伸到机身/腿上——换到侧向锚点时
    # 候选真的会落到基座后面（实测 aim x=-0.40）。默认 0 = 只扫前半平面；给负值放开。
    "aim_forward_min_m": 0.0,
}

# ---- estimate_surface_z 的取样几何 ----
# 这些**不进配置**：它们跟着相机装法和画面布局走，不是现场能调的参数。
# 画面最下面 20% 是夹爪手指（真机那帧里手指就在这儿，深度是 0 或者手指自己），
# 拿它估地面会把高度估成爪子的高度。左右各留 15% 是躲开边缘畸变和画面外扩的残影。
_SURFACE_ROW_RANGE = (0.15, 0.80)
_SURFACE_COL_RANGE = (0.15, 0.85)
# 先取 z 的低分位当"支撑面大概多高"：物体/手指/臂在画面里都是**高于**支撑面的东西，
# 低分位天然落在支撑面这一侧（板子/地面）。25% 是标定那边"板面取点"同样用的比例。
_SURFACE_LOW_PERCENTILE = 25.0
# 再在低分位附近 ±3 cm 取点求中位数：低分位是"这一侧"，中位数才是"这个面"。
# 3 cm 覆盖 D435i 在 1 m 处的深度噪声（实测 ±1 cm 量级），又不会把 10 cm 高的物体带进来。
_SURFACE_BAND_M = 0.03
# 有效点太少说明这一帧基本没量到东西（或者全 0），估出来的数不可信——宁可返回 None，
# 让调用方退回场景先验/干脆不扫。
_SURFACE_MIN_POINTS = 200
# 深度图里明显不是测量值的大数（D435i 的无效值有时是 65535，不是 0）按无效处理。
_SURFACE_MAX_DEPTH_MM = 20000.0

# 关节限位两侧留的余量（度）。URDF 的限位是"理论可达"，贴边走过去伺服会顶住/报警，
# 和 handeye_auto 里留 3° 是同一个道理，这里只要 1°（观察位不是极限动作）。
_LIMIT_MARGIN_DEG = 1.0


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _as_float(value, name):
    """配置里的标量统一成 float：yaml 写 4 和 4.0 是同一个东西，但 4.0 参与算比较不会
    突然变整数除法，也不会把 numpy 的 int/float 混用带进后面的判断。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("observe_scan 配置 %s 要一个数，收到 %r" % (name, value))


def _as_bool(value):
    """开关统一成 bool。yaml 给的是布尔，但从环境变量/命令行来的时候可能是
    "false"/"0" 这种字符串——按字符串真值判断它们全是 True，必须单独认。"""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _as_float_pair(value, name):
    """两元区间统一成 [下限, 上限] 的 list[float]。

    顺手排序：调用方把 [1.6, 0.3] 写反了也照样能用，比抛错友善，也免得"配置写反了
    就永远选不出位姿"这种查半天的问题。
    """
    try:
        numbers = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError("observe_scan 配置 %s 要两个数 [下限, 上限]，收到 %r" % (name, value))
    if numbers.size != 2:
        raise ValueError("observe_scan 配置 %s 要两个数 [下限, 上限]，收到 %r" % (name, value))
    return [float(min(numbers)), float(max(numbers))]


def _as_spread_levels(value):
    """撒点档位统一成 [[6 个数], ...]（度）。

    允许只写一档（`[12, 8, 8, 18, 15, 25]`）——现场调参时常常只想改一档试试，
    为这个在 yaml 里套两层列表太啰嗦。
    """
    try:
        levels = list(value)
    except TypeError:
        raise ValueError("observe_scan 配置 spread_levels_deg 要是档位列表，收到 %r" % (value,))
    if not levels:
        raise ValueError("observe_scan 配置 spread_levels_deg 不能是空列表")
    if all(not isinstance(level, (list, tuple, np.ndarray)) for level in levels):
        levels = [levels]                       # 单档简写
    normalized = []
    for level in levels:
        try:
            numbers = np.asarray(level, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError("observe_scan 配置 spread_levels_deg 每档要 6 个数，收到 %r" % (level,))
        if numbers.size != 6:
            raise ValueError("observe_scan 配置 spread_levels_deg 每档要 6 个数，收到 %r" % (level,))
        normalized.append([float(v) for v in numbers])
    return normalized


def _as_anchors(value):
    """额外观察锚点统一成 [[6 个数], ...]（度）。

    **空列表/null 都合法**（= 只用标称位）。写错任何一组就让整份配置报错而不是静默丢掉：
    锚点是"人指定要走到哪儿"的强约束，少走一组会让人以为扫过了、其实没去。
    """
    if value is None:
        return []
    try:
        groups = list(value)
    except TypeError:
        raise ValueError("observe_scan 配置 anchors_deg 要是锚点列表，收到 %r" % (value,))
    normalized = []
    for order, group in enumerate(groups, start=1):
        try:
            numbers = np.asarray(group, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError("observe_scan 配置 anchors_deg 第 %d 组要 6 个数，收到 %r"
                             % (order, group))
        if numbers.size != 6:
            raise ValueError("observe_scan 配置 anchors_deg 第 %d 组要 6 个数，收到 %r"
                             % (order, group))
        if not np.all(np.isfinite(numbers)):
            raise ValueError("observe_scan 配置 anchors_deg 第 %d 组里有非有限数：%r"
                             % (order, group))
        normalized.append([float(v) for v in numbers])
    return normalized


def merged_config(config=None) -> dict:
    """默认值 + 用户覆盖，数值统一成 float / list[float]。

    **只认 SCAN_DEFAULTS 里的键**，其它键一律忽略、不报错：调用方很可能把整份技能配置
    递进来（里面还有 ip、tcp、夹爪尺寸……），为几个不认识的键把整次扫描毙掉不值得。
    config 不是字典（None、或者上游传了个坏的）时按"全用默认值"处理——多观察位扫描是
    补救路径，不是关键路径，不该因为它把整轮抓取搞崩。

    Returns:
        新字典（深拷贝），调用方改它不会污染 SCAN_DEFAULTS。
    """
    merged = copy.deepcopy(SCAN_DEFAULTS)
    if isinstance(config, dict):
        for key in SCAN_DEFAULTS:
            if key in config and config[key] is not None:
                merged[key] = config[key]
    merged["enabled"] = _as_bool(merged["enabled"])
    for key in ("max_views", "samples_per_level", "seed"):
        merged[key] = _as_float(merged[key], key)
    for key in ("max_deviation_deg", "min_camera_height_m", "min_tip_height_m",
                "min_rotation_gap_deg", "min_aim_gap_m", "aim_forward_min_m"):
        merged[key] = _as_float(merged[key], key)
    for key in ("distance_m", "view_angle_deg", "aim_radius_m"):
        merged[key] = _as_float_pair(merged[key], key)
    merged["spread_levels_deg"] = _as_spread_levels(merged["spread_levels_deg"])
    merged["anchors_deg"] = _as_anchors(merged["anchors_deg"])
    return merged


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _rotation_gap_deg(a, b) -> float:
    """两个姿态之间差多少度（把旋转差折成一个转轴一个角）。

    和 pipeline.orientation_gap_deg 是同一个式子，但**不能 import pipeline**：
    pipeline 会 import 本文件，反过来 import 就成环了。三行，抄一遍不心痛。
    """
    cosine = (np.trace(np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def joint_limits_deg(calib) -> np.ndarray:
    """这份 calib 的关节限位，[(lo, hi)] × 6，**度**。

    两种来源都认，与任务里定下的鸭子类型一致：
      - 真 Calibration：`.bounds` 是 URDF 原生弧度；
      - 假对象/别的臂：可能只给 `.joint_limits_deg()`（度）。
    两样都没有就没法规划（夹不了限位，撒出来的位姿可能根本走不了），直接报错比瞎走强。
    """
    bounds = getattr(calib, "bounds", None)
    if bounds is not None:
        limits = np.rad2deg(np.asarray(bounds, dtype=float).reshape(-1, 2))
    else:
        getter = getattr(calib, "joint_limits_deg", None)
        if getter is None:
            raise ValueError("calib 既没有 .bounds 也没有 .joint_limits_deg()，没法夹关节限位")
        limits = np.asarray(getter(), dtype=float).reshape(-1, 2)
    if limits.shape != (6, 2):
        raise ValueError("关节限位形状不对：要 6×2，收到 %s" % (limits.shape,))
    return limits


def _camera_pose(calib, joints_deg):
    """关节角 → 相机在基座系的位姿 4×4。包一层只是统一 reshape 一次。"""
    return np.asarray(calib.camera(joints_deg), dtype=float).reshape(4, 4)


def _tip_in_base(calib, joints_deg):
    """关节角 → 爪尖在基座系的位置。

    要走一次法兰的正运动学，比相机多一次 FK。所以**只在候选已经过了前面几条约束、
    真的需要判"爪子会不会扎到桌面"时才调**——撒 600 个点，每个都走两遍 FK 是白花时间。
    """
    flange = np.asarray(calib.flange(joints_deg), dtype=float).reshape(4, 4)
    tip_in_flange = np.asarray(calib.tip_in_flange, dtype=float).reshape(3)
    return flange[:3, 3] + flange[:3, :3] @ tip_in_flange


# ---------------------------------------------------------------------------
# 支撑面高度先验
# ---------------------------------------------------------------------------

def estimate_surface_z(depth_mm, camera_pose, camera_matrix, config=None) -> float | None:
    """从**已经拍到的那一帧深度**里估一个"支撑面高度"（基座系 z，米）。

    这里刻意不做检测/分割：规划只需要一个"东西大概落在多高的平面上"的先验，用来算
    "光轴还要往下走多远、爪尖够不够高"。真机上这个数拿不到（配置里的 table_height_m
    已经过时），但从刚拍到的那帧深度里现估是稳的——相机反正已经拍过了，不用多走一步。

    步骤：
      1. 只取画面**中上部**（避开最下面 20% 的夹爪手指），把有效深度点反投影到基座系；
      2. 取基座系 z 的 25% 低分位——画面里手指、物体、臂都在支撑面**之上**，低分位
         天然落在支撑面这一侧；
      3. 在低分位 ±3 cm 内取点求 z 中位数：低分位只是"大概"，中位数才是这个面。

    Args:
        depth_mm: H×W 深度图，单位毫米（D435i 的深度流原样给的就是 uint16 毫米）。
            兼容误传成米的输入（整帧都 < 20 显然不是毫米，自动 ×1000）——这条只是兜底，
            免得调用方单位写错时估出一个天上/地下的数还照样拿去规划。
        camera_pose: 4×4，相机在**基座系**的位姿（cam → base，和 pipeline 的 state["camera_pose"] 同向）。
        camera_matrix: 3×3 内参（fx, fy, cx, cy）。
        config: 只为和 plan_views 共用一份配置字典才留的参数；估面没有现场可调项（见
            文件顶部 _SURFACE_* 的说明），所以这里不读它。

    Returns:
        支撑面高度（基座系 z，米）；这一帧有效点太少/全 0/形状不对时返回 None。
    """
    del config                       # 估面没有可调项，见 docstring
    depth = np.asarray(depth_mm, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        return None
    try:
        pose = np.asarray(camera_pose, dtype=float).reshape(4, 4)
        camera = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    except (TypeError, ValueError):
        return None

    height, width = depth.shape
    row_lo = max(0, min(height - 1, int(round(height * _SURFACE_ROW_RANGE[0]))))
    row_hi = max(row_lo + 1, min(height, int(round(height * _SURFACE_ROW_RANGE[1]))))
    col_lo = max(0, min(width - 1, int(round(width * _SURFACE_COL_RANGE[0]))))
    col_hi = max(col_lo + 1, min(width, int(round(width * _SURFACE_COL_RANGE[1]))))
    region = depth[row_lo:row_hi, col_lo:col_hi]

    valid = np.isfinite(region) & (region > 0.0) & (region < _SURFACE_MAX_DEPTH_MM)
    if int(valid.sum()) < _SURFACE_MIN_POINTS:
        return None
    rows, cols = np.nonzero(valid)
    depth_values = region[rows, cols]
    # 单位兜底：整帧深度中位数 < 20 只可能是"米"，不可能是"毫米"（毫米下 20 mm 内
    # 根本不在 D435i 的量程里）。统一成毫米之后转米——**下面反投影必须用米**：
    # camera_pose 的平移是米，拿毫米去加会差 1000 倍。
    if float(np.median(depth_values)) < 20.0:
        depth_values = depth_values * 1000.0
    depth_m = depth_values * 0.001

    fx = float(camera[0, 0])
    fy = float(camera[1, 1])
    cx = float(camera[0, 2])
    cy = float(camera[1, 2])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        return None
    # 针孔反投影（深度是沿光轴的 z，不是斜距）：p_cam = z * [(u-cx)/fx, (v-cy)/fy, 1]。
    # 不套畸变系数：深度图和彩色已经对齐/校正过，1 m 量级上畸变对"取个平面高度"没影响。
    u = cols.astype(np.float64) + col_lo
    v = rows.astype(np.float64) + row_lo
    points_cam = np.stack([(u - cx) * depth_m / fx,
                           (v - cy) * depth_m / fy,
                           depth_m], axis=1)
    points_base = points_cam @ pose[:3, :3].T + pose[:3, 3]

    z_base = points_base[:, 2]
    low = float(np.percentile(z_base, _SURFACE_LOW_PERCENTILE))
    near = z_base[np.abs(z_base - low) <= _SURFACE_BAND_M]
    if near.size >= max(30, _SURFACE_MIN_POINTS // 4):
        estimate = float(np.median(near))
    else:
        # 低分位附近点太少（画面里几乎全是斜面/杂物）：退回"低分位这一侧所有点"的中位数
        # ——仍然比直接拿低分位当答案稳，因为单点噪声不会直接把结果带偏。
        estimate = float(np.median(z_base[z_base <= low + _SURFACE_BAND_M]))
    if not np.isfinite(estimate):
        return None
    return float(round(estimate, 4))


# ---------------------------------------------------------------------------
# 多观察位规划
# ---------------------------------------------------------------------------

def plan_views(calib, nominal_joints_deg, surface_z, config=None) -> dict:
    """在标称观察位附近撒点，挑几个"彼此差异大、又都合格"的备用观察位。

    每个候选位姿 q 都要过这几关（全部在基座系，z 朝上）：
      1. 先夹进关节限位（留 1° 余量），且相对标称位姿的单关节偏移 ≤ max_deviation_deg；
      2. 光轴朝下：axis[2] < 0，视角 θ = acos(-axis_z) 在 view_angle_deg 内；
      3. 光轴与支撑面 z=surface_z 的交点离相机 distance_m，落点水平半径在 aim_radius_m
         （臂基座在原点，落点跑出可达范围就过去也够不着），且**落在基座前方**
         （aim[0] ≥ aim_forward_min_m，默认只扫前半平面——物体只在前方，后面够不着）；
      4. 相机离面 ≥ min_camera_height_m，爪尖离面 ≥ min_tip_height_m（安全，别扎桌面）。
    这个函数只围绕传入的 nominal_joints_deg 撒点，**不读 anchors_deg**：多锚点的循环在
    pipeline 里，每个锚点各调一次 plan_views。换锚点 = 换一个关节角向量，别的不用改。
    合格的候选里再做最远点采样：以标称 aim 为初始种子，每次挑"离标称+已选 aim 的水平
    距离最远"的，且它的相机旋转与标称+已选至少差 min_rotation_gap_deg；挑满 max_views
    或者最好的候选也近于 min_aim_gap_m 就停。最后按 deviation 升序——小动作先走，稳。

    Args:
        calib: 鸭子类型标定对象：`.camera(q)` / `.flange(q)` → 4×4（基座系），
            `.tip_in_flange` 是 3 维（米），`.bounds`（弧度）或 `.joint_limits_deg()`
            （度）给关节限位（`.X` = T_flange_camera 是这套接口的一部分，本模块不直接用，
            相机位姿一律问 camera()）。
        nominal_joints_deg: 标称拍照位的 6 个关节角（度），真机就是 calib.observe_joints。
        surface_z: 支撑面高度（基座系 z，米）。由调用方从深度帧估（estimate_surface_z）
            或退回场景先验；**不使用**配置里的 table_height_m。
        config: 覆盖 SCAN_DEFAULTS 的字典，见 merged_config。

    Returns:
        {"ok": bool, "views": [...], "notes": [...], "nominal": {...}, "config": {...}}
        - views 每项：joints_deg[6], aim_m[3], distance_m, view_angle_deg,
          deviation_deg, rotation_gap_deg, reason（给人看的一句中文）；
        - ok=False 时 views 为空、notes 非空（写清卡在哪一步）；
        - nominal 是标称位自己的落点/距离/视角（规划时拿它当"初始种子"）；
        - config 是本次实际生效的配置（合并 + 归一化之后）。
    """
    try:
        cfg = merged_config(config)
    except (TypeError, ValueError) as error:
        # 配置写坏（比如 distance_m 只给了一个数）不该让整轮抓取崩：多观察位是补救路径，
        # 不扫就是了，但原因要如实带回去，别静默用默认值继续走。
        return {"ok": False, "views": [],
                "notes": ["observe_scan 配置不合法，不规划：%s" % error],
                "nominal": {}, "config": {}}
    notes: list[str] = []
    result = {"ok": False, "views": [], "notes": notes, "nominal": {}, "config": cfg}

    if not cfg["enabled"]:
        notes.append("多观察位扫描已关（enabled=false）：不规划，停在标称拍照位。")
        return result
    max_views = int(cfg["max_views"])
    if max_views < 1:
        notes.append("max_views=%d < 1：不规划多观察位。" % max_views)
        return result
    try:
        surface = float(surface_z)
    except (TypeError, ValueError):
        notes.append("surface_z 不是数（%r），没有支撑面高度先验，不规划。" % (surface_z,))
        return result
    if not np.isfinite(surface):
        notes.append("surface_z 不是有限数（%r），不规划。" % (surface_z,))
        return result
    nominal_joints = np.asarray(nominal_joints_deg, dtype=float).reshape(-1)
    if nominal_joints.size != 6:
        notes.append("标称位姿要有 6 个关节角，收到 %d 个，不规划。" % nominal_joints.size)
        return result

    limits = joint_limits_deg(calib)
    clip_lo = limits[:, 0] + _LIMIT_MARGIN_DEG
    clip_hi = limits[:, 1] - _LIMIT_MARGIN_DEG
    # 限位本身比 2×余量还窄的关节（正常 URDF 不会出现）夹到中点，别造出 lo>hi 的区间。
    middle = (limits[:, 0] + limits[:, 1]) / 2.0
    clip_lo = np.minimum(clip_lo, middle)
    clip_hi = np.maximum(clip_hi, middle)
    outside = [index + 1 for index in range(6)
               if not (clip_lo[index] <= nominal_joints[index] <= clip_hi[index])]
    if outside:
        # 真机就是这个情况：observe_joints 的 J5=69.99°，而 URDF 限位是 69.9°。
        # 标称位是标定教出来的、实际能站住，所以照它算 nominal 和撒点中心；但候选一律
        # 夹回限位再筛——备用位姿是**新走**过去的，得是限位内走得动的姿态。
        notes.append("注意：标称位姿的 J%s 超出「限位 + 1° 余量」的可用区间，候选会先夹回限位再筛。"
                     % "、J".join(str(index) for index in outside))

    # 标称位自己的几何：报告里给出、挑选时当"初始种子"。
    nominal_camera = _camera_pose(calib, nominal_joints)
    nominal_axis = nominal_camera[:3, 2]
    nominal_position = nominal_camera[:3, 3]
    if nominal_axis[2] >= -1e-9:
        notes.append("标称位姿的光轴不朝下（axis_z=%.3f），拿不到支撑面高度先验，不规划。"
                     % nominal_axis[2])
        return result
    if nominal_position[2] <= surface:
        notes.append("标称位的相机已经在支撑面以下（z=%.3f ≤ surface_z=%.3f），没有朝下的观察可言。"
                     % (nominal_position[2], surface))
        return result
    nominal_distance = float((surface - nominal_position[2]) / nominal_axis[2])
    nominal_aim = nominal_position + nominal_distance * nominal_axis
    nominal_view_angle = float(np.rad2deg(np.arccos(np.clip(-nominal_axis[2], -1.0, 1.0))))
    result["nominal"] = {
        "aim_m": [float(round(value, 4)) for value in nominal_aim],
        "distance_m": float(round(nominal_distance, 3)),
        "view_angle_deg": float(round(nominal_view_angle, 1)),
    }
    notes.append(
        "标称位：相机在 [%.3f, %.3f, %.3f]，光轴 [%.3f, %.3f, %.3f]，aim=[%.3f, %.3f, %.3f]，"
        "离面 %.3f m，视角 %.1f°。"
        % (nominal_position[0], nominal_position[1], nominal_position[2],
           nominal_axis[0], nominal_axis[1], nominal_axis[2],
           nominal_aim[0], nominal_aim[1], nominal_aim[2],
           nominal_distance, nominal_view_angle))
    notes.append("支撑面取 surface_z=%.3f m（调用方从深度帧估的；不用配置里过时的 table_height_m）。"
                 % surface)

    # ---- 撒点 + 逐条筛 ----
    seed = int(cfg["seed"])
    if seed < 0:
        # numpy 不认负数种子；取绝对值就行——"确定性"要的是同一份配置出同一批样本，
        # 不是"种子必须唯一"。
        seed = -seed
    per_level = int(cfg["samples_per_level"])
    distance_range = cfg["distance_m"]
    view_range = cfg["view_angle_deg"]
    aim_range = cfg["aim_radius_m"]
    max_deviation = cfg["max_deviation_deg"]
    min_camera_height = cfg["min_camera_height_m"]
    min_tip_height = cfg["min_tip_height_m"]
    aim_forward_min = cfg["aim_forward_min_m"]
    rng = np.random.default_rng(seed)
    # 拒绝计数按"卡在哪一条"分开记：真机调参时只看"没选出来"没法下手，得知道是哪条闸在拦。
    rejects = {"deviation": 0, "axis_up": 0, "view_angle": 0, "distance": 0,
               "aim_radius": 0, "aim_forward": 0, "camera_height": 0, "tip_height": 0}
    clamped = 0
    candidates: list[dict] = []
    accepted_per_level: list[int] = []
    total = 0
    for level_index, spread in enumerate(cfg["spread_levels_deg"]):
        spread = np.asarray(spread, dtype=float)
        accepted = 0
        for _ in range(per_level):
            total += 1
            joints = nominal_joints + rng.uniform(-1.0, 1.0, 6) * spread
            clamped_joints = np.clip(joints, clip_lo, clip_hi)
            if np.any(np.abs(clamped_joints - joints) > 1e-9):
                clamped += 1
            joints = clamped_joints
            deviation = float(np.max(np.abs(joints - nominal_joints)))
            if deviation > max_deviation:
                rejects["deviation"] += 1
                continue
            camera = _camera_pose(calib, joints)
            axis = camera[:3, 2]
            position = camera[:3, 3]
            if axis[2] >= 0.0:
                rejects["axis_up"] += 1
                continue
            view_angle = float(np.rad2deg(np.arccos(np.clip(-axis[2], -1.0, 1.0))))
            if not (view_range[0] <= view_angle <= view_range[1]):
                rejects["view_angle"] += 1
                continue
            # 光轴与平面 z=surface 的交点参数 t（沿光轴的距离，米）。axis[2]<0 且
            # 相机在面上方，所以 t 一定是正的，不用再判符号。
            distance = float((surface - position[2]) / axis[2])
            if not (distance_range[0] <= distance <= distance_range[1]):
                rejects["distance"] += 1
                continue
            aim = position + distance * axis
            aim_radius = float(np.hypot(aim[0], aim[1]))
            if not (aim_range[0] <= aim_radius <= aim_range[1]):
                rejects["aim_radius"] += 1
                continue
            # 基座系 +x 是臂的正前方：要抓的物体只会在前方。换到侧向锚点时，撒出来的
            # 候选会有一部分落到基座后面（实测 aim x=-0.40）——走过去也够不着/被机身挡住。
            if aim[0] < aim_forward_min:
                rejects["aim_forward"] += 1
                continue
            if position[2] - surface < min_camera_height:
                rejects["camera_height"] += 1
                continue
            tip = _tip_in_base(calib, joints)
            if tip[2] - surface < min_tip_height:
                rejects["tip_height"] += 1
                continue
            accepted += 1
            candidates.append({"joints": joints, "camera": camera, "aim": aim,
                               "distance": distance, "view_angle": view_angle,
                               "deviation": deviation, "level": level_index})
        accepted_per_level.append(accepted)

    notes.append("撒点 %d 个（%d 档 × %d 个/档）：合格 %d 个。"
                 % (total, len(cfg["spread_levels_deg"]), per_level, len(candidates)))
    notes.append("各档合格数：%s。"
                 % "，".join("第 %d 档 %d 个（spread=[%s]°）"
                            % (index + 1, count,
                               ", ".join("%g" % value for value in spread))
                            for index, (spread, count) in enumerate(
                                zip(cfg["spread_levels_deg"], accepted_per_level))))
    notes.append(
        "拒绝计数：关节偏差 >%.1f° %d；光轴不朝下 %d；视角越界 %d；离面距离越界 %d；"
        "aim 半径越界 %d；落点跑到基座后方（x<%.2f m）%d；相机太低 %d；爪尖太低 %d；"
        "另有 %d 个样本先被夹回限位。"
        % (max_deviation, rejects["deviation"], rejects["axis_up"], rejects["view_angle"],
           rejects["distance"], rejects["aim_radius"], aim_forward_min,
           rejects["aim_forward"], rejects["camera_height"], rejects["tip_height"], clamped))

    # ---- 最远点采样挑位姿 ----
    min_rotation_gap = cfg["min_rotation_gap_deg"]
    min_aim_gap = cfg["min_aim_gap_m"]
    anchors = [nominal_aim[:2].copy()]          # aim 的 XY 种子：标称位自己
    used_rotations = [nominal_camera[:3, :3].copy()]
    chosen: list[int] = []
    chosen_set: set[int] = set()
    rotation_blocked: set[int] = set()
    stop_reason = ""
    while len(chosen) < max_views:
        best_index = -1
        best_aim_gap = -1.0
        best_rotation_gap = 0.0
        for index, candidate in enumerate(candidates):
            if index in chosen_set:
                continue
            rotation_gap = min(
                _rotation_gap_deg(candidate["camera"][:3, :3], rotation)
                for rotation in used_rotations)
            if rotation_gap < min_rotation_gap:
                rotation_blocked.add(index)
                continue
            aim_gap = min(float(np.linalg.norm(candidate["aim"][:2] - anchor))
                          for anchor in anchors)
            if aim_gap > best_aim_gap:
                best_index = index
                best_aim_gap = aim_gap
                best_rotation_gap = rotation_gap
        if best_index < 0:
            stop_reason = ("没有相机旋转差 ≥%.1f° 的候选了" % min_rotation_gap
                           if candidates else "一个合格候选都没有")
            break
        if best_aim_gap < min_aim_gap:
            stop_reason = ("最好的候选 aim 离标称/已选也只有 %.3f m < %.2f m，"
                           "再挪也换不出新视角" % (best_aim_gap, min_aim_gap))
            break
        candidate = candidates[best_index]
        candidate["rotation_gap"] = best_rotation_gap   # 选中那一刻的差（对"标称+已选"取最小）
        candidate["aim_gap"] = best_aim_gap
        candidate["selection_order"] = len(chosen)
        chosen.append(best_index)
        chosen_set.add(best_index)
        anchors.append(candidate["aim"][:2].copy())
        used_rotations.append(candidate["camera"][:3, :3].copy())
    if not stop_reason:
        stop_reason = "选满 max_views=%d 个（到上限）" % max_views
    if rotation_blocked:
        notes.append("挑位姿时因相机旋转差 <%.1f° 跳过 %d 个候选（换角度不够大，去了也白去）。"
                     % (min_rotation_gap, len(rotation_blocked)))
    notes.append("最终选 %d 个：%s。" % (len(chosen), stop_reason))

    # 按 deviation 升序：先走动作小的（离标称近），走位风险小、真机也更快。
    picked = sorted((candidates[index] for index in chosen), key=lambda item: item["deviation"])
    views = []
    for candidate in picked:
        joints = [float(round(value, 3)) for value in candidate["joints"]]
        aim = [float(round(value, 4)) for value in candidate["aim"]]
        views.append({
            "joints_deg": joints,
            "aim_m": aim,
            "distance_m": float(round(candidate["distance"], 3)),
            "view_angle_deg": float(round(candidate["view_angle"], 1)),
            "deviation_deg": float(round(candidate["deviation"], 1)),
            "rotation_gap_deg": float(round(candidate["rotation_gap"], 1)),
            "reason": ("换角度看：aim 离标称/已选 %.2f m、相机转了 %.1f°、关节最多偏 %.1f°"
                       "（视角 %.1f°）"
                       % (candidate["aim_gap"], candidate["rotation_gap"],
                          candidate["deviation"], candidate["view_angle"])),
        })
    result["views"] = views
    result["ok"] = bool(views)
    if not views:
        notes.append("没有值得走的备用观察位，不扫（不回退到「随便走一个」——那只是多花时间）。")
    return result
