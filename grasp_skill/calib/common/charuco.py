# Description: ChArUco 标定板的识别、位姿求解与内参标定，全部走 OpenCV 4.11 的新 aruco 接口。
#
# 为什么用 ChArUco 而不是纯棋盘格：纯棋盘格必须整块拍全，一角出画就整张废掉；ChArUco 每个
# 格子里印了一个 aruco 码，认出几个码就能定位到具体是哪几个角点，遮住一半照样能用。臂举着
# 相机贴近看的时候板子经常出画，这个性质很关键。
#
# 注意 OpenCV 4.11 只有新接口：CharucoDetector / CharucoBoard / getPredefinedDictionary。
# 老接口（CharucoBoard_create、interpolateCornersCharuco、calibrateCameraCharuco、
# estimatePoseCharucoBoard）在 4.11 里已经删了，网上抄来的老代码跑不起来。
import cv2
import numpy as np

# 板子上印的是哪套码，买来的板子说明书上会写；认不出来就用 identify_dictionary() 扫一遍
DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class Board:
    """一块 ChArUco 板：格子数、格子边长、码边长、用哪套码。

    squares_x / squares_y 数的是黑白格子数，不是内角点数。内角点数是 (squares_x-1) x
    (squares_y-1)，买板子的规格表一般写格子数。square_len_m 与 marker_len_m 必须拿尺子
    量实物，单位是米：这两个数直接决定解出来的位姿是多少米，量错 1% 位姿就差 1%。
    """

    def __init__(self, squares_x: int, squares_y: int, square_len_m: float,
                 marker_len_m: float, dictionary: str = "DICT_5X5_100"):
        if marker_len_m >= square_len_m:
            raise ValueError(f"码边长 {marker_len_m} 必须小于格子边长 {square_len_m}")
        if dictionary not in DICTIONARIES:
            raise ValueError(f"不认识的码表 {dictionary}，可选：{list(DICTIONARIES)}")
        self.squares_x = int(squares_x)
        self.squares_y = int(squares_y)
        self.square_len_m = float(square_len_m)
        self.marker_len_m = float(marker_len_m)
        self.dictionary_name = dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[dictionary])
        self.board = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_len_m,
                                            self.marker_len_m, self.dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    @property
    def total_corners(self) -> int:
        """板子一共有多少个内角点。"""
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def size_m(self) -> tuple[float, float]:
        """板子的长宽，单位米。"""
        return self.squares_x * self.square_len_m, self.squares_y * self.square_len_m

    def corner_points(self) -> np.ndarray:
        """所有内角点在板子自身坐标系里的三维坐标，N x 3，z 全为 0。"""
        return self.board.getChessboardCorners().astype(np.float64)

    def describe(self) -> str:
        width, height = self.size_m
        return (f"{self.squares_x}x{self.squares_y} 格，格子 {self.square_len_m * 1000:.1f} mm，"
                f"码 {self.marker_len_m * 1000:.1f} mm，{self.dictionary_name}，"
                f"共 {self.total_corners} 个内角点，板面 {width * 1000:.0f}x{height * 1000:.0f} mm")

    def detect(self, image) -> tuple[np.ndarray | None, np.ndarray | None]:
        """在一张图里找内角点。

        Args:
            image: BGR 或灰度图。

        Returns:
            (corners, ids)：corners 是 N x 1 x 2 的亚像素坐标，ids 是 N x 1 的角点编号；
            找不到时两个都是 None。
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if corners is None or ids is None or len(ids) == 0:
            return None, None
        return corners, ids

    def draw(self, image, corners, ids):
        """把找到的角点画在图上，返回新图。"""
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if corners is not None and ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(canvas, corners, ids, (0, 255, 0))
        return canvas


def identify_dictionary(image, min_markers: int = 4) -> list[dict]:
    """扫一遍所有码表，看哪套认得出这块板子上的码。

    买来的板子说明书上不一定写用的是哪套码，而码表选错一个角点都检不出。这里对每套码跑一遍
    纯 aruco 检测，跟格子数、边长都无关，所以规格还没填对的时候也能用。

    同一个位宽的几套码是包含关系（DICT_5X5_50 的 50 个图案就是 DICT_5X5_100 的前 50 个），
    所以编号都在 50 以内的板子会被 5X5 那一族全部认出来，扫描能定到族、定不到族里具体哪套。
    这不影响结果：认出的图案一样，解出来的角点也一样。个数一样多时取最小的那套，
    编号超出范围的会少认几个，个数自然会把它排下去。

    Args:
        image: BGR 或灰度图。
        min_markers: 认出的码少于这个数就当没认出，避免把误检当成结果。

    Returns:
        [{"dictionary", "count", "max_id", "size"}, ...]，认出个数从多到少；没认出的不列。
        max_id 是认出的码里编号最大的那个，看全整块板子时它跟格子数对得上：
        ChArUco 每两格印一个码，所以格子数约等于 2 x (max_id + 1)。
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = []
    for name, code in DICTIONARIES.items():
        dictionary = cv2.aruco.getPredefinedDictionary(code)
        _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray)
        if ids is None or len(ids) < min_markers:
            continue
        found.append({"dictionary": name, "count": int(len(ids)), "max_id": int(ids.max()),
                      "size": int(dictionary.bytesList.shape[0])})
    found.sort(key=lambda item: (-item["count"], item["size"]))
    return found


def estimate_pose(board: Board, corners, ids, camera_matrix, dist_coeffs,
                  min_corners: int = 6) -> tuple[np.ndarray | None, float, str]:
    """解标定板在相机系里的位姿 T_camera_board。

    板子是平面，所以用 IPPE 求初值再迭代精化：平面 PnP 在这上面比默认的迭代法稳。

    Args:
        board: 板子规格。
        corners: detect() 返回的角点。
        ids: detect() 返回的角点编号。
        camera_matrix: 3x3 内参。
        dist_coeffs: 畸变系数；图已经去过畸变时传全零。
        min_corners: 少于这么多角点就不解，点太少解出来的位姿不可信。

    Returns:
        (T_camera_board 或 None, 重投影误差像素, 说明文字)。
    """
    if corners is None or ids is None or len(ids) < min_corners:
        found = 0 if ids is None else len(ids)
        return None, float("inf"), f"只认出 {found} 个角点，少于 {min_corners} 个，不解位姿"
    object_points = board.corner_points()[ids.flatten()]
    image_points = corners.reshape(-1, 2).astype(np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, float("inf"), "solvePnP 没解出位姿"
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, dist_coeffs,
                                      rvec, tvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    distance = float(np.linalg.norm(tvec))
    return transform, error, f"{len(ids)} 个角点，距离 {distance * 1000:.0f} mm，重投影误差 {error:.3f} px"


def viewing_angle_deg(transform) -> float:
    """相机光轴与板面法线的夹角，单位度。

    这个角太小（正对着拍）时板子的倾斜量测不准，太大（贴着桌面斜看）时角点会糊。
    30° 到 60° 之间最好。
    """
    normal = np.asarray(transform, dtype=float)[:3, 2]
    axis = np.array([0.0, 0.0, 1.0])
    cosine = abs(float(np.dot(normal, axis)))
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def calibrate_intrinsics(board: Board, detections: list, image_size: tuple,
                         fix_principal_point: bool = True) -> dict:
    """拿多张 ChArUco 照片标内参。

    detections 是 [(corners, ids), ...]。这里用的是与 common/chessboard.py 同一个思路：
    先把主点钉在图像中心、切向畸变按零来解，避免焦距与畸变互相抵消解出一组好看但错的值；
    再放开精化，只有焦距没跑飞、误差也没变大时才采纳精化的结果。

    Args:
        board: 板子规格。
        detections: 每张图的 (corners, ids)，角点少于 6 个的会被丢掉。
        image_size: (宽, 高)。
        fix_principal_point: 第一阶段是否把主点钉在图像中心。

    Returns:
        {"camera_matrix", "dist_coeffs", "rms", "used", "stage"}。

    Raises:
        ValueError: 可用的照片少于 4 张。
    """
    object_points, image_points = [], []
    for corners, ids in detections:
        if corners is None or ids is None or len(ids) < 6:
            continue
        object_points.append(board.corner_points()[ids.flatten()].astype(np.float32))
        image_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
    if len(object_points) < 4:
        raise ValueError(f"能用的照片只有 {len(object_points)} 张，少于 4 张标不出内参")
    width, height = image_size
    guess = np.array([[max(width, height), 0, width / 2.0],
                      [0, max(width, height), height / 2.0], [0, 0, 1]], dtype=np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    rms1, matrix1, dist1, _, _ = cv2.calibrateCamera(object_points, image_points, (width, height),
                                                     guess.copy(), None, flags=flags)
    rms2, matrix2, dist2, _, _ = cv2.calibrateCamera(object_points, image_points, (width, height),
                                                     matrix1.copy(), dist1.copy(),
                                                     flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    focal_sane = 0.6 * matrix1[0][0] <= matrix2[0][0] <= 1.6 * matrix1[0][0]
    if focal_sane and rms2 <= rms1 * 1.05:
        return {"camera_matrix": matrix2, "dist_coeffs": dist2, "rms": float(rms2),
                "used": len(object_points), "stage": "放开精化"}
    return {"camera_matrix": matrix1, "dist_coeffs": dist1, "rms": float(rms1),
            "used": len(object_points), "stage": "钉住主点与切向畸变"}


def compare_intrinsics(factory, calibrated) -> str:
    """把工厂内参与自己标出来的内参摆在一起比。

    深度相机出厂就带内参，自己再标一遍主要是为了核对。两边差得多说明有一边不对，
    要先查清楚再往下走，不能随手选一个。
    """
    factory = np.asarray(factory, dtype=float).reshape(3, 3)
    calibrated = np.asarray(calibrated, dtype=float).reshape(3, 3)
    names = [("fx", (0, 0)), ("fy", (1, 1)), ("cx", (0, 2)), ("cy", (1, 2))]
    lines = ["            工厂      自标      差值      相对差"]
    for name, (i, j) in names:
        a, b = factory[i][j], calibrated[i][j]
        relative = (b - a) / a * 100 if abs(a) > 1e-9 else float("nan")
        lines.append(f"  {name:<8}{a:9.2f}{b:10.2f}{b - a:10.2f}{relative:9.2f}%")
    focal_diff = abs(calibrated[0][0] - factory[0][0]) / factory[0][0] * 100
    if focal_diff > 3.0:
        lines.append(f"  焦距差了 {focal_diff:.1f}%，超过 3%。先查板子的格子边长量得对不对，"
                     "再查照片是不是都从差不多的距离拍的。")
    else:
        lines.append(f"  焦距差 {focal_diff:.1f}%，在 3% 以内，两边一致。")
    return "\n".join(lines)
