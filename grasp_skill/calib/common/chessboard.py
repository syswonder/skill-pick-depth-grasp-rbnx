# Description: 棋盘格检测与相机内参求解——两个部署共用（纯 OpenCV，不碰硬件）。
#
# 采集由各自的 app 负责（浏览器点一下存一张），这里只管"算"。
import cv2
import numpy as np


def detect_chessboard_points(color_image: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """检测棋盘格内角点并做亚像素细化；没检测到完整棋盘格时返回 None。"""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found or corners is None:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined_corners.reshape(-1, 2).astype(np.float32)


def make_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    """棋盘格角点的世界坐标。

    square_size 只影响解出来的棋盘位姿（每张图的 tvec）的尺度：把物点整体放大 s 倍，
    K/D 一模一样，只是 tvec 也放大 s 倍。所以内参标定填错格子尺寸没关系，
    真要拿相机量实际距离时才依赖它。
    """
    """棋盘格角点在标定板坐标系下的理论坐标（z=0 的平面点阵）。"""
    cols, rows = pattern_size
    points = []
    for row in range(rows):
        for col in range(cols):
            points.append([col * square_size, row * square_size, 0.0])
    return np.array(points, dtype=np.float32)


def calibrate_from_images(image_paths: list[str], pattern_size: tuple[int, int],
                          square_size: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """由采集到的图片求解内参。

    Returns:
        (内参矩阵 K, 畸变系数 D, 总体重投影误差 RMS（像素）)。

    Raises:
        RuntimeError: 有效图片不足 3 张。
    """
    objp = make_object_points(pattern_size, square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图片: {image_path}，已跳过")
            continue
        corners = detect_chessboard_points(image, pattern_size)
        if corners is None:
            print(f"图片 {image_path} 未检测到完整棋盘格，已跳过")
            continue
        object_points.append(objp)
        image_points.append(corners)
        image_size = (image.shape[1], image.shape[0])

    if image_size is None or len(object_points) < 3:
        raise RuntimeError("有效标定图片不足，至少需要3张检测到棋盘格的图片")

    # 这批照片量得出焦距吗？量不出就别让焦距乱跑（钉住它），报出来的数才稳定
    scan_rms = [r for _, r in _focal_rms_scan(object_points, image_points, image_size)]
    free_focal = max(scan_rms) - min(scan_rms) >= 0.05
    camera_matrix, dist_coeffs, rms = _calibrate_robust(object_points, image_points,
                                                        image_size, free_focal)
    return camera_matrix, dist_coeffs, float(rms)


def _calibrate_robust(object_points: list, image_points: list, image_size: tuple[int, int],
                      free_focal: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """两段式解内参：先给"正常镜头"的初值解一遍，再以它为初值精修。

    为什么不直接 cv2.calibrateCamera(K=None)：焦距和畸变有一处等价的坑——"焦距超大 + 畸变超大"
    与"焦距正常 + 畸变小"在同一组照片上的重投影误差几乎一样（实测 0.612 vs 0.623 px，画面只差几个像素），
    从无初值起步会落进前面那个坑：fx 能离谱到十几倍，误差却很好看。第一段用
    fx=fy=max(w,h)、主点居中、只放开 k1/k2，把解带回正常那一支；第二段放开全部参数精修，
    精修要是跑偏（焦距超出合理范围或误差变差）就退回第一段的结果。

    free_focal=False（照片量不出焦距时）直接跳过第二段的焦距自由度：反正怎么解都一样准，
    钉住比让它随机漂到一个等价解上更好交代。
    """
    w, h = image_size
    guess = np.array([[max(w, h), 0.0, w / 2.0],
                      [0.0, max(w, h), h / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    stage1_flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
                    | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    rms1, K1, D1, _, _ = cv2.calibrateCamera(object_points, image_points, image_size,
                                             guess.copy(), None, flags=stage1_flags)
    if free_focal:
        rms2, K2, D2, _, _ = cv2.calibrateCamera(object_points, image_points, image_size,
                                                 K1.copy(), D1.copy(),
                                                 flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        # 只在初值附近有意义的范围内采纳：跑太远说明本来就没约束住，退回钉住焦距的那一版
        if 0.6 * K1[0][0] <= K2[0][0] <= 1.6 * K1[0][0] and rms2 <= rms1 * 1.05:
            return K2, D2, float(rms2)
    stage2b_flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH)
    rms3, K3, D3, _, _ = cv2.calibrateCamera(object_points, image_points, image_size,
                                             K1.copy(), D1.copy(), flags=stage2b_flags)
    return K3, D3, float(rms3)


def _collect_corners(image_paths: list[str], pattern_size: tuple[int, int]):
    """把每张图检测到的角点收成 (物点, 像点, 图像尺寸)。"""
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        corners = detect_chessboard_points(image, pattern_size)
        if corners is None:
            continue
        object_points.append(make_object_points(pattern_size, 1.0))
        image_points.append(corners)
        size = (image.shape[1], image.shape[0])
    return object_points, image_points, size


def _focal_rms_scan(object_points, image_points, size) -> list[tuple[float, float]]:
    """把焦距钉在一串固定值上各解一遍，返回 [(fx, RMS)]。

    焦距和畸变会互相顶班：板子在画面里小、姿态又单调时，fx 差十几倍的两组参数在同一批照片上
    几乎一样准（实测 0.543~0.554 px）。误差不随焦距变，就说明这批照片量不出焦距。
    """
    w, h = size
    out: list[tuple[float, float]] = []
    for f in (0.5 * w, 0.75 * w, w, 1.5 * w, 2.5 * w, 4 * w):
        guess = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH
                 | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
        rms, *_ = cv2.calibrateCamera(object_points, image_points, size, guess, None, flags=flags)
        out.append((float(f), float(rms)))
    return out


def focal_uncertainty_warning(image_paths: list[str], pattern_size: tuple[int, int]) -> str | None:
    """这批照片量得出焦距吗——用"钉住焦距扫一遍、看误差变不变"来量。"""
    object_points, image_points, size = _collect_corners(image_paths, pattern_size)
    if size is None or len(object_points) < 5:
        return None
    scan = _focal_rms_scan(object_points, image_points, size)
    rms_values = [r for _, r in scan]
    lo, hi = min(rms_values), max(rms_values)
    if hi - lo >= 0.05:
        return None
    f_lo, f_hi = scan[0][0], scan[-1][0]
    return (f"⚠️ 这批照片量不出焦距：把 fx 钉在 {f_lo:.0f}~{f_hi:.0f}（{f_hi / f_lo:.0f} 倍范围）里逐点解，"
            f"重投影误差只在 {lo:.3f}~{hi:.3f} px 之间变——焦距被畸变顶班了，报出来的值只是众多等价解之一。"
            "重拍：板子拿近些（占画面 1/4 以上）、每次明显倾斜 ±20~30°、远近也变一变，采 15 张以上再解。")


def compute_per_view_errors(image_paths: list[str], pattern_size: tuple[int, int],
                            camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> list[float]:
    """逐张图片的重投影误差（像素），用来看是哪几张拖了后腿。"""
    objp = make_object_points(pattern_size, 1.0)  # 误差与格子实际尺寸无关
    errors: list[float] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        corners = detect_chessboard_points(image, pattern_size)
        if corners is None:
            continue
        found, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        if not found:
            continue
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        difference = corners.reshape(-1, 2) - projected.reshape(-1, 2)
        error = np.sqrt(np.mean(np.sum(difference**2, axis=1)))
        errors.append(float(error))
    return errors
