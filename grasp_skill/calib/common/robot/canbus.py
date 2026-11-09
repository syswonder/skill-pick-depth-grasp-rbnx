# Description: 找机械臂的 CAN 口——和臂原语一样的办法（本子树里只有这一份）。
#
# 臂原语 piper_ctl/main.py 的 _find_gs_usb_can()：USB-CAN 适配器（gs_usb）的网卡在
# sysfs 里指向 USB 设备，而 Jetson 板载的 mttcan 指向 platform/...；接口名开机后可能
# 会漂（can0/can4/...），所以按驱动拓扑认口，而不是写死一个名字。
#
# 探测是只读的，起停是**显式**的：口没起来（DOWN）时只打印提示，绝不自动去改接口状态
# （别的进程可能正在用它）；页面上那两个按钮点下去才动，见 set_up / set_down。
import glob
import os
import re
import subprocess

# Piper 走的就是这个波特率（同臂原语 can_activate.sh 的默认值）。
BITRATE = 1000000


def list_can_interfaces() -> list[str]:
    """当前系统里所有 CAN 口（含板载 mttcan），已排序。"""
    return sorted(os.path.basename(p) for p in glob.glob("/sys/class/net/can*"))


def find_usb_can() -> str | None:
    """USB-CAN 适配器的接口名（如 "can4"）；找不到返回 None。"""
    for path in sorted(glob.glob("/sys/class/net/can*")):
        try:
            dev = os.path.realpath(os.path.join(path, "device"))
        except OSError:
            continue
        if "usb" in dev.lower():
            return os.path.basename(path)
    return None


def is_up(iface: str) -> bool:
    """接口是否已 UP（读 sysfs，不需要权限）。"""
    try:
        with open(f"/sys/class/net/{iface}/operstate", encoding="utf-8") as f:
            return f.read().strip() in ("up", "unknown")
    except OSError:
        return False


def _ip_output(args: list[str], timeout_s: float = 5.0) -> str:
    """跑一条只读的 ip 命令，返回 stdout；跑不动就返回空串（页面上显示成"-"）。"""
    try:
        done = subprocess.run(["ip"] + args, capture_output=True, text=True, timeout=timeout_s)
    except Exception:  # noqa: BLE001
        return ""
    return done.stdout


def bitrate_of(iface: str) -> str:
    """接口当前波特率（读不出来就返回空）——只看，不改。

    照 `bitrate <n>` 抓，别按"最长的数字"猜：没设过波特率时 `ip -details` 里还有
    clock 48000000 这样的数，猜出来的是时钟频率（踩过）。
    """
    match = re.search(r"\bbitrate (\d+)", _ip_output(["-details", "link", "show", iface]))
    return match.group(1) if match else ""


def bus_state(iface: str) -> str:
    """CAN 总线状态：STOPPED（口停着）、ERROR-ACTIVE（正常收发）、BUS-OFF（总线掉了）。"""
    match = re.search(r"\bcan state (\S+)", _ip_output(["-details", "link", "show", iface]))
    return match.group(1) if match else ""


def port_state(iface: str) -> dict:
    """口现在什么样：在不在、UP/DOWN、波特率、总线状态。只读，不碰硬件。

    页面上"CAN 口"那一行靠它——口没起来时，这一行是唯一能看出毛病在哪儿的线索
    （臂那边什么读数都给不出来：连不上就是连不上）。
    """
    state = {"iface": iface, "exists": False, "up": False, "bitrate": "", "bus": ""}
    if not os.path.exists(f"/sys/class/net/{iface}"):
        return state
    state["exists"] = True
    state["up"] = is_up(iface)
    state["bitrate"] = bitrate_of(iface)
    state["bus"] = bus_state(iface)
    return state


# 提权 helper：仓库里带的那两个脚本（robot/arm-can-up、arm-can-down），装到
# /usr/local/sbin 配一行 NOPASSWD sudoers——见 tools/calibration/README.md。
# 只放行"起停这一个 CAN 口"，比给整个网页服务 root 窄得多。
HELPER_DIR = "/usr/local/sbin"


def _run_helper(name: str) -> str | None:
    """调提权 helper（sudo -n <HELPER_DIR>/<name>）。返回 None = 成功，否则一句人话。"""
    path = os.path.join(HELPER_DIR, name)
    if not os.path.exists(path):
        return "没装提权 helper（%s）——装法见 tools/calibration/README.md" % path
    try:
        done = subprocess.run(["sudo", "-n", path], capture_output=True, text=True, timeout=30)
    except Exception as error:  # noqa: BLE001
        return "跑 %s 失败：%s" % (path, error)
    if done.returncode == 0:
        return None
    return (done.stderr or done.stdout).strip() or "helper 返回 %d" % done.returncode


def _try_ip(args: list[str], prefix: list[str], timeout_s: float) -> str | None:
    """跑一条改状态的 ip 命令；成功返回 None，失败返回那句错误。"""
    command = prefix + ["ip"] + args
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except Exception as error:  # noqa: BLE001
        return "跑 %s 失败：%s" % (" ".join(command), error)
    if done.returncode == 0:
        return None
    return (done.stderr or done.stdout).strip() or "ip 返回 %d" % done.returncode


def _run_ip(args: list[str], timeout_s: float = 10.0, manual: str = "") -> str | None:
    """改接口状态：先直接跑，没权限再试免密 sudo。返回 None = 成功，否则是一句人话。

    标定台是普通用户起的（不是 root），而改接口状态要 CAP_NET_ADMIN。两条路都不通时
    把该手敲的命令原样报出来——只说"权限不足"等于把人丢回 shell，报命令才是这页该做的。
    报的是 `manual`（整条命令链）：只报失败的这一步没用，那个口多半得先 down 再设波特率。
    """
    direct = _try_ip(args, [], timeout_s)
    if direct is None or "permitted" not in direct.lower():
        return direct
    elevated = _try_ip(args, ["sudo", "-n"], timeout_s)
    if elevated is None:
        return None
    return ("改接口状态要 root：当前用户没权限，免密 sudo 也不可用（%s）。"
            "在宿主上手敲：%s" % (elevated, manual or ("sudo ip " + " ".join(args))))


def set_up(iface: str, bitrate: int = BITRATE) -> str | None:
    """把口拉起来：设波特率 → up。返回 None = 成功，否则是错误说明。

    本来就是 UP 且波特率对，就一步都不发——正在用的口被弹一下，会话就断了。
    """
    if is_up(iface) and bitrate_of(iface) == str(bitrate):
        return None
    # 波特率只能在口停着的时候改，所以先 down 一次（本来就没起来的话，这步是无害的）
    manual = ("sudo ip link set %s down && sudo ip link set %s type can bitrate %d "
              "&& sudo ip link set %s up" % (iface, iface, bitrate, iface))
    if iface == find_usb_can():
        # 这个口就是 helper 认的那个：直接让它整段做完（down → 波特率 → up），
        # 页面就不必握着通用 sudo
        if _run_helper("arm-can-up") is None:
            return None
    for args in (["link", "set", iface, "down"],
                 ["link", "set", iface, "type", "can", "bitrate", str(bitrate)],
                 ["link", "set", iface, "up"]):
        error = _run_ip(args, manual=manual)
        if error:
            return error
    return None


def set_down(iface: str) -> str | None:
    """把口停掉。返回 None = 成功，否则是错误说明。"""
    if not is_up(iface):
        return None
    if iface == find_usb_can():
        if _run_helper("arm-can-down") is None:
            return None
    return _run_ip(["link", "set", iface, "down"],
                   manual="sudo ip link set %s down" % iface)


def resolve_arm_port(configured: str | None = None, verbose: bool = True) -> str:
    """定下用哪个 CAN 口。

    顺序是有意的：**先自动探测**（同臂原语），探测不到才退回配置值——接口名会漂，
    配置里那个名字多半是上一次开机抄的。

    Args:
        configured: 配置里的 `arm_port`（可为空/None）。
        verbose: 是否打印探测结果与提示。

    Returns:
        接口名，例如 "can4"。

    Raises:
        RuntimeError: 既没探测到 USB-CAN，配置里也没给口名。
    """
    detected = find_usb_can()
    if detected:
        if verbose:
            if configured and configured != detected:
                print(f"CAN 口自动探测：用 {detected}（配置里写的是 {configured}，以探测结果为准）", flush=True)
            else:
                print(f"CAN 口自动探测：用 {detected}", flush=True)
            if not is_up(detected):
                print(f"⚠️ {detected} 当前是 DOWN——先把它起起来再用。"
                      f"标定台「臂控制」页上那个「拉起 CAN」就是干这个的；"
                      f"命令行等价于：\n"
                      f"    sudo ip link set {detected} type can bitrate 1000000 "
                      f"&& sudo ip link set {detected} up", flush=True)
        return detected

    if configured:
        if verbose:
            print(f"没探测到 USB-CAN（gs_usb），退回配置里的 arm_port={configured}；"
                  f"现有 CAN 口: {list_can_interfaces() or '无'}", flush=True)
        return configured

    raise RuntimeError(
        "没找到 USB-CAN 接口，配置里也没写 arm_port。"
        f"现有 CAN 口: {list_can_interfaces() or '无'}（板载 mttcan 不算，要的是 USB-CAN 适配器）"
    )
