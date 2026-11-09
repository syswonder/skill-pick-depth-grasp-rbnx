# Description: 抓取位姿与下爪深度的解算。纯数学，不碰硬件、不读图像。
#
# 输入的数是：物体顶面上一点、它所在平面的法线、物体长轴与闭合方向（面内的两个单位
# 向量）、它压在平面上的投影宽度。这四样除平面法线外都由**视觉那侧**从图像算好交过来
# （见视觉服务 service/perception/grasp_geometry）。输出的数是：悬停点、下爪终点（爪尖该停在哪儿）、
# 法兰姿态，加上一串"这个物体这套夹爪抓不抓得了"的判断。
#
# **本文件不碰像素、不收内参外参**：原来 `plan()` 里有两处要把检测框投影到三维
# （长轴方向、投影宽度），那两处已经搬进视觉服务了。
#
# 姿态怎么定：接近方向是平面法线的反方向（垂直于平面往下走），闭合方向落在平面内、
# 并且垂直于物体的长轴（从长边两侧夹，而不是顶着长轴推）。法兰的 z 轴是接近方向，
# 夹爪两指的运动方向是法兰的 y 轴（见 piper 的 URDF：joint7/joint8 沿法兰 -y 平移，
# 单指行程 35 mm），所以两列一定，第三列由右手系定。
#
# 下爪多深由三个约束夹出来，见 solve_descent()。爪尖到法兰那 142 mm 含连接件，
# 真正能夹住物体的只有"可用夹持段"。
import numpy as np

# 悬停高度：物体顶面沿法线往上这么多
HOVER_M = 0.10
# 指尖离平面的余量：这几毫米是留给深度噪声和平面拟合误差的
TIP_CLEARANCE_M = 0.004
# 指根（两指之间的那块本体）离物体顶面的余量
ROOT_CLEARANCE_M = 0.005
# 抓取点离指尖的余量：别把物体卡在指尖上
TIP_MARGIN_M = 0.005
# 抓取高度取物体高度的这个比例（一半，拦腰夹最稳）
GRASP_HEIGHT_RATIO = 0.5
# 抓取高度的上下限：太靠近顶面或底面的抓法都不稳
GRASP_HEIGHT_MIN_M = 0.005
GRASP_TOP_MARGIN_M = 0.003
# 物体在抓取高度处的宽度要小于开口减这个余量
OPENING_MARGIN_M = 0.010
# 超过这个高度就不当"平放的小物体"处理了，多半是平面拟合选错了面
MAX_OBJECT_HEIGHT_M = 0.20

# 允许的姿态偏差。J5 只有 ±69.9°，斜面一大，"严格垂直于平面"就转不过去。实际按能到多少
# 算多少，偏差记在计划里；偏到超过这个角度就不要了——那样两指是斜着切进去的，
# 接触从面变成线，抓不稳。实测这条臂：绕基座 y 轴倾 40° 时正好用满，绕 x 轴约 20°。
MAX_OBLIQUITY_DEG = 12.0
# 到位误差分两个方向看：横向差多少、沿接近方向差多少。沿轴深一点浅一点无所谓，
# 物体只是落在夹持段的不同位置；横向差了就是夹在边上或者夹空。
LATERAL_TOL_M = 0.005
ALONG_TOL_M = 0.010
# 闭合方向要落在平面内：偏出平面就是两指一个在上一个在下地夹，等于用指尖顶着物体
CLOSING_OFF_PLANE_TOL_DEG = 8.0


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("方向向量的长度是零")
    return vector / norm


def in_plane_basis(normal):
    """平面内的两个正交单位方向，用来枚举绕法线的转角。"""
    normal = _unit(normal)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(helper @ normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(helper, normal)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def candidate_closings(plan, step_deg: float = 30.0, limit_deg: float = 90.0) -> list:
    """按优先级排出可以试的闭合方向。

    物体长轴决定的首选是"垂直于长轴"，但手腕不一定转得到那个方向——绕法线换方向等价于
    换手腕的转角，J5/J6 的限位直接卡在这上面（实测：同样竖直朝下，有的转角 0 mm 到位，
    有的差 20 mm 以上）。所以从首选开始左右交替往外扩，逐个试，谁先到就用谁。

    Returns:
        [(单位方向, 相对首选的转角偏差度), ...]，按偏差从小到大。
    """
    normal = _unit(plan["plane"]["normal"])
    first, second = in_plane_basis(normal)
    preferred = _unit(plan["closing"])
    start = float(np.arctan2(preferred @ second, preferred @ first))
    out = []
    steps = int(limit_deg / step_deg)
    for index in range(steps + 1):
        offsets = (0.0,) if index == 0 else (index * step_deg, -index * step_deg)
        for offset in offsets:
            angle = start + np.deg2rad(offset)
            direction = np.cos(angle) * first + np.sin(angle) * second
            out.append((_unit(direction), abs(offset)))
    return out


# 下爪那一步允许差多少：**从爪和物体自己的尺寸算，不拍常数**。
# 两指能不能把物体夹在中间，只取决于横向差有没有超过 (开口 − 物体宽)/2 ——
# 超过它，两指就落在物体的同一侧，合爪夹空；没超过，两指仍然跨在物体两侧。
# 留 30% 余量：物体宽度本身是量出来的，有误差。
# 上限 20 mm：物体很窄时上面那条会放到几十毫米，那已经不叫"对准"了。
DESCEND_FIT_RATIO = 0.7
DESCEND_SETTLED_MAX_M = 0.020


def descend_settled_m(gripper: dict, footprint_width_m: float | None) -> float:
    """下爪那一步"算到位"的判据（米）。见上面常量那段。"""
    if not footprint_width_m:
        return DESCEND_SETTLED_MAX_M
    margin = max(0.0, (float(gripper["opening_mm"]) / 1000.0 - float(footprint_width_m)) / 2.0)
    return min(DESCEND_FIT_RATIO * margin, DESCEND_SETTLED_MAX_M)


def rotation_from_axes(approach, closing) -> np.ndarray:
    """法兰姿态：z 轴是接近方向，y 轴是两指运动方向（闭合方向）。

    piper 的 URDF 里 joint7/joint8 沿法兰 -y 平移，所以两指的开合方向就是法兰的 y 轴。
    正负号无所谓，夹爪是对称的。
    """
    z_axis = _unit(approach)
    y_axis = _unit(closing)
    y_axis = y_axis - float(y_axis @ z_axis) * z_axis          # 正交化
    if float(np.linalg.norm(y_axis)) < 1e-9:
        raise ValueError("闭合方向与接近方向平行")
    y_axis = _unit(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def solve_descent(object_height_m, clamp_length_m) -> dict:
    """解爪尖该停在离平面多高。

    三个约束（都沿法线量，从平面往上为正）：
        t >= 指尖余量                      爪尖不能扎进平面
        t + L >= 物体高度 + 指根余量        两指之间那块本体不能压到物体顶面
        t <= 抓取高度 - 指尖上余量          抓取点要落在夹持段内、且离指尖有点距离

    Returns:
        {"ok", "height_m" 或 None, "lower_m", "upper_m", "why"}
    """
    # 抓取点：一般取半高（重心附近，稳）——**但不能低于"指根刚好不压物体顶面"那条线**。
    # 手指够不到半高时（物体比指长还高）就得把抓取点往上抬：实测 131 mm 高的纸巾、
    # 指长 72 mm，半高 65.4 mm 够不到（指根会先压在包顶上），抬到 68.8 mm 才可行。
    # 上限仍是"物体顶面往下留 GRASP_TOP_MARGIN"——所以矮物体那条"太矮"的判据不受影响
    # （10 mm 高的东西抬到 7 mm，还是过不了 4 vs 2 那条）。
    lower = max(TIP_CLEARANCE_M, object_height_m + ROOT_CLEARANCE_M - clamp_length_m)
    desired = min(max(object_height_m * GRASP_HEIGHT_RATIO, GRASP_HEIGHT_MIN_M),
                  max(object_height_m - GRASP_TOP_MARGIN_M, GRASP_HEIGHT_MIN_M))
    grasp_height = min(max(desired, lower + TIP_MARGIN_M),
                       max(object_height_m - GRASP_TOP_MARGIN_M, GRASP_HEIGHT_MIN_M))
    upper = grasp_height - TIP_MARGIN_M
    if lower > upper + 1e-9:
        if object_height_m + ROOT_CLEARANCE_M - clamp_length_m > TIP_CLEARANCE_M:
            why = ("可用夹持段 %.0f mm 太短：物体高 %.0f mm，指根会先压在物体顶面上。"
                   "换个更高的抓取位置，或者换更长的指。" % (clamp_length_m * 1000, object_height_m * 1000))
        else:
            why = ("物体太矮（%.0f mm）：抓取高度减去余量之后，爪尖已经低于平面的安全高度。"
                   % (object_height_m * 1000))
        return {"ok": False, "height_m": None, "lower_m": lower, "upper_m": upper,
                "grasp_height_m": grasp_height, "why": why}
    return {"ok": True, "height_m": 0.5 * (lower + upper), "lower_m": lower, "upper_m": upper,
            "grasp_height_m": grasp_height, "why": ""}


def plan(top_point, plane, long_axis, closing, footprint_width,
         gripper: dict, hover_m: float = HOVER_M) -> dict:
    """把一次抓取的位姿算出来。

    纯三维：**不收像素、不收内参外参**。物体朝哪（`long_axis`）、两指在哪儿合
    （`closing`）、它压在平面上多宽（`footprint_width`）这三样是**视觉那侧**从图像算好
    交过来的（见视觉服务 service/perception/grasp_geometry）。所以这个函数不碰图像，跟硬件也无关。

    Args:
        top_point: 物体顶面上一点（米），与 `plane` 同一个坐标系。
        plane: 支撑平面 {"point", "normal"}，法线指向平面外侧。
        long_axis: 顶面内的单位方向（物体长轴）。同一个坐标系。
        closing: 闭合方向（面内、垂直于长轴）。同一个坐标系。
        footprint_width: 物体**顶面**在闭合方向上的跨度（米）；量不出来给 None。
            （视觉那侧从深度里分出顶面那一层、量它在闭合方向的跨度，不是拿框去量。）
        gripper: {"clamp_length_mm", "opening_mm", "closed_gap_mm"}。

    Returns:
        计划字典；`ok` 为假时 `problems` 里逐条写原因。
    """
    problems = []
    normal = _unit(plane["normal"])
    top_point = np.asarray(top_point, dtype=float).reshape(3)
    long_axis = _unit(long_axis)
    closing = _unit(closing)
    object_height = float((top_point - np.asarray(plane["point"], dtype=float)) @ normal)
    if object_height <= 0:
        problems.append("物体顶面落在拟合出来的平面之下（%.1f mm）：要么平面选错了，要么这一眼的"
                        "物体顶面量错了（框里没有物体的深度、拿旁边的桌面顶了包）。"
                        % (object_height * 1000))
    elif object_height > MAX_OBJECT_HEIGHT_M:
        problems.append("物体高出平面 %.0f mm，超过 %.0f mm 就当它不是平放的小物体，"
                        "先确认拟合到的是不是它所在的那个面。"
                        % (object_height * 1000, MAX_OBJECT_HEIGHT_M * 1000))

    clamp_length = float(gripper["clamp_length_mm"]) / 1000.0
    opening = float(gripper["opening_mm"]) / 1000.0
    descent = solve_descent(max(object_height, 0.0), clamp_length)
    if not descent["ok"]:
        problems.append(descent["why"])

    approach = -normal
    rotation = rotation_from_axes(approach, closing)

    if footprint_width is not None and footprint_width > opening - OPENING_MARGIN_M:
        problems.append("物体在闭合方向上宽约 %.0f mm，开口只有 %.0f mm（还要留 %.0f mm 余量），"
                        "夹不住。" % (footprint_width * 1000, opening * 1000, OPENING_MARGIN_M * 1000))

    hover_tip = top_point + hover_m * normal
    tip_height = descent["height_m"]
    grasp_tip = None
    descent_distance = None
    if tip_height is not None:
        grasp_tip = top_point + (tip_height - object_height) * normal
        descent_distance = float((hover_tip - grasp_tip) @ normal)

    return {
        "ok": not problems,
        "problems": problems,
        "plane": plane,
        "object_height_m": object_height,
        "grasp_height_m": descent.get("grasp_height_m"),
        "tip_height_m": tip_height,
        "descent_m": descent_distance,
        "hover_tip": hover_tip.tolist(),
        "grasp_tip": None if grasp_tip is None else grasp_tip.tolist(),
        "approach": approach.tolist(),
        "closing": closing.tolist(),
        "long_axis": long_axis.tolist(),
        "rotation": rotation.tolist(),
        "footprint_width_m": footprint_width,
        "bounds_m": {"lower": descent["lower_m"], "upper": descent["upper_m"]},
    }


def evaluate_reach(plan, solved, max_obliquity_deg: float = MAX_OBLIQUITY_DEG,
                   closing_tol_deg: float = CLOSING_OFF_PLANE_TOL_DEG,
                   lateral_tol_m: float | None = None,
                   along_tol_m: float | None = None) -> dict:
    """到位之后，实际姿态离"垂直于平面"差了多少、爪尖差了多远。

    IK 是带着目标姿态去解的，斜面大到一定程度就解不到那个姿态，只能给一个最接近的。
    与其一刀切说不合格，不如把差量算出来：只要两指还是落在平面内合拢（不是斜着切进去）、
    横向没偏、沿轴深浅无所谓，这一爪照样能夹住。

    Returns:
        {"ok", "reasons", "obliquity_deg", "lateral_miss_m", "along_miss_m",
         "closing_off_plane_deg"}
    """
    # 横向/沿轴两个容差做成参数：悬停只是**路过**，这两条在它那儿是冗余的（下爪那一步
    # 是按臂实到的关节角重新解 IK 的，悬停偏几毫米会被它吃掉）。抓取那一步用默认值。
    lateral_tol = LATERAL_TOL_M if lateral_tol_m is None else lateral_tol_m
    along_tol = ALONG_TOL_M if along_tol_m is None else along_tol_m
    reasons = []
    rotation = np.asarray(solved["rotation"], dtype=float)
    tip = np.asarray(solved["tip"], dtype=float)
    target_tip = np.asarray(solved.get("target_tip", plan["grasp_tip"]), dtype=float)
    normal = _unit(plan["plane"]["normal"])
    approach = -normal

    tool_axis = rotation[:, 2]
    obliquity = float(np.degrees(np.arccos(np.clip(float(tool_axis @ approach), -1.0, 1.0))))
    if obliquity > max_obliquity_deg:
        reasons.append("姿态偏 %.1f°，超过允许的 %.0f°：这条斜面对这副腕来说太陡了。"
                       % (obliquity, max_obliquity_deg))

    closing = rotation[:, 1]
    off_plane = float(np.degrees(np.arcsin(np.clip(abs(float(closing @ normal)), -1.0, 1.0))))
    if off_plane > closing_tol_deg:
        reasons.append("闭合方向偏出平面 %.1f°（上限 %.0f°），两指会一个高一个低地夹。"
                       % (off_plane, closing_tol_deg))

    miss = tip - target_tip
    along = float(miss @ tool_axis)
    lateral = float(np.linalg.norm(miss - along * tool_axis))
    if lateral > lateral_tol:
        reasons.append("横向差 %.1f mm（上限 %.0f mm），可能夹在物体的边上。"
                       % (lateral * 1000, lateral_tol * 1000))
    if abs(along) > along_tol:
        reasons.append("沿接近方向差 %.1f mm（上限 %.0f mm），爪尖没停在该停的高度。"
                       % (abs(along) * 1000, along_tol * 1000))

    return {"ok": not reasons, "reasons": reasons, "obliquity_deg": obliquity,
            "lateral_miss_m": lateral, "along_miss_m": along,
            "closing_off_plane_deg": off_plane}


