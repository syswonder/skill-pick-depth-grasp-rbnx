# Description: 相机与臂的唯一持有者——一个进程里只该有一条相机 pipeline 和一条 CAN 会话。
#
# 两个标定台共用：相机类由各部署注入（S10 是 pyrealsense2，piper 是 pyorbbecsdk），
# 臂是一样的（同一个臂、同一套 CAN 指令，见 common/robot/direct.py）。
#
# 浏览器上可能同时开着几页，而相机/臂是独占硬件：所以这里做"占用"——步骤要用就先
# acquire()，别的步骤进来会拿到一句明确的报错（页面上显示谁占着），用完 release()。
import threading

from common.camera import UndistortCamera
from common.robot.canbus import resolve_arm_port
from common.robot.direct import DirectArm


class Rig:
    def __init__(self, camera_factory, paths, port: str | None = None):
        """
        Args:
            camera_factory: 无参可调用，返回一个相机对象（有 get_frames/close）。
            paths: 这个部署的产物路径模块（去畸变要读里面的内参文件）。
            port: CAN 口；`None` = 自动探测（同臂原语）。
        """
        self._camera_factory = camera_factory
        self._paths = paths
        self._port = port
        self._lock = threading.Lock()
        self._owner: str | None = None
        self._cam = None
        self._cam_error: str | None = None   # 打开相机失败的原因（页面上要看得到）
        self._cam_undist: UndistortCamera | None = None
        self._arm: DirectArm | None = None

    # ---------- 占用 ----------

    def acquire(self, owner: str, force: bool = False) -> None:
        """声明占用；被别的步骤占着就抛 RuntimeError（页面把这句话显示出来）。

        Args:
            owner: 步骤名。
            force: `True` = 页面上点了「强制接管」，把别的步骤挤掉。
        """
        with self._lock:
            if self._owner and self._owner != owner and not force:
                raise RuntimeError(f"「{self._owner}」正在用相机/臂，先到那个页面点「结束本步骤」。")
            self._owner = owner

    def release(self, owner: str) -> None:
        """释放硬件：关相机、断臂。"""
        with self._lock:
            if self._owner not in (None, owner):
                raise RuntimeError(f"「{self._owner}」正占着，不能由「{owner}」释放。")
            self._owner = None
            self._close_hardware()

    def owner(self) -> str | None:
        return self._owner

    # ---------- 硬件（懒建） ----------

    def has_intrinsics(self) -> bool:
        return self._paths.CAMERA_MATRIX.exists() and self._paths.DIST_COEFFS.exists()

    def camera(self, raw: bool = False):
        """取相机。

        Args:
            raw: `True` = 原始画面（内参标定用——去畸变就看不出畸变了）；
                `False` = 有内参就去畸变（手眼标定、抓取自检要与运行时一致）。
        """
        with self._lock:
            if self._cam is None:
                try:
                    self._cam = self._camera_factory()
                    self._cam_error = None
                except Exception as e:  # noqa: BLE001
                    self._cam_error = f"打开相机失败：{type(e).__name__}: {e}"
                    raise
            if raw:
                return self._cam
            if self._cam_undist is None and self.has_intrinsics():
                self._cam_undist = UndistortCamera(self._cam, self._paths.CAMERA_MATRIX,
                                                   self._paths.DIST_COEFFS)
            return self._cam_undist or self._cam

    def arm(self) -> DirectArm:
        """取臂会话（直连 CAN）。"""
        with self._lock:
            if self._arm is None:
                self._arm = DirectArm(port=self._port)
            return self._arm

    def arm_port(self) -> str:
        """这个部署实际会用哪个 CAN 口（与 DirectArm 同一套判据，不连臂）。

        口没起来时臂那边什么都读不出来，"哪个口、起来没有"只能靠这个。
        """
        return self._port or resolve_arm_port(verbose=False)

    def drop_arm(self) -> None:
        """断开臂会话（相机不动）——要换成别的臂后端时用（同一个 CAN 口不该有两条连接）。"""
        with self._lock:
            if self._arm is not None:
                self._arm.close()
                self._arm = None

    def _close_hardware(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None
            self._cam_undist = None
        if self._arm is not None:
            self._arm.close()
            self._arm = None

    # ---------- 状态 ----------

    def status(self) -> dict:
        return {
            "owner": self._owner,
            "camera_open": self._cam is not None,
            "arm_open": self._arm is not None,
            "arm_port": self._port or "自动探测",
            "has_intrinsics": self.has_intrinsics(),
            "camera_error": self.camera_error(),
        }

    def camera_error(self) -> str | None:
        """相机最近一次出错的原因（打开失败、或取帧一直失败）；正常时为 None。

        页面靠它把"画面出不来"变成一句能读懂的话——相机类自己把异常吞了，
        不这么兜一层，前端只会看到一个转圈的空白。
        """
        return self._cam_error or getattr(self._cam, "last_error", None)
