# Description: 两个标定台的公共界面层——页面外壳 + 步骤基类（Flask）。
#
# 一个"步骤"就是标定流程里的一步（内参、手眼…）：一页 + 一组动作接口 + 一段产物摘要。
# app.py 只负责把步骤挂起来；每步内部怎么走，由步骤自己写在 steps/ 里。
from flask import Response, jsonify

from common.camera import mjpeg_stream

STYLE = """<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Noto Sans SC", sans-serif;
       background: #14161a; color: #e6e6e6; margin: 0; }
header { display: flex; align-items: center; gap: 18px; padding: 12px 20px;
         background: #1c1f24; border-bottom: 1px solid #2b3038; flex-wrap: wrap; }
header .brand { font-weight: 600; color: #fff; text-decoration: none; }
nav a { color: #9fb0c0; text-decoration: none; margin-right: 12px; font-size: 14px; }
nav a.on { color: #4fd1a5; font-weight: 600; }
main { max-width: 1180px; margin: 0 auto; padding: 20px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 20px 0 8px; }
.desc { color: #9aa4b0; font-size: 14px; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }
.card { background: #1c1f24; border: 1px solid #2b3038; border-radius: 8px; padding: 16px; }
.card h3 { margin: 0 0 6px; font-size: 16px; }
.card .lines { font-size: 13px; color: #9aa4b0; white-space: pre-wrap; min-height: 3em; }
.card a { color: #4fd1a5; text-decoration: none; }
.tag { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #2b3038; color: #9fb0c0; }
.tag.ok { background: #17402f; color: #6ee7b7; }
.row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
.col { flex: 1 1 420px; min-width: 320px; }
img.stream { width: 100%; background: #000; border: 1px solid #2b3038;
             border-radius: 6px; display: block; cursor: crosshair; }
button { padding: 9px 16px; font-size: 14px; margin: 0 8px 8px 0; cursor: pointer;
         border: 1px solid #37404b; border-radius: 6px; background: #232830; color: #e6e6e6; }
button:hover { background: #2b3038; }
button.primary { background: #1f6f54; border-color: #2a8a6a; color: #fff; }
button.danger { background: #5a2226; border-color: #7d3037; color: #ffd9db; }
button:disabled { opacity: .5; cursor: not-allowed; }
input, select { padding: 7px 10px; background: #232830; color: #e6e6e6;
                border: 1px solid #37404b; border-radius: 6px; }
.status { margin: 12px 0; padding: 10px 14px; background: #1c1f24; border-left: 4px solid #f0a94a;
          border-radius: 6px; font-size: 14px; white-space: pre-wrap; }
.status.ok { border-left-color: #4fd1a5; }
.status.err { border-left-color: #d9534f; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px;
        background: #101216; padding: 10px 12px; border-radius: 6px; overflow-x: auto;
        white-space: pre-wrap; color: #b8c4d0; }
.kv { font-size: 13px; color: #9aa4b0; line-height: 1.7; }
.kv b { color: #e6e6e6; font-weight: 600; }
</style>"""

# 硬件占用提示 + 两个按钮：所有页面共用（谁占着相机/臂、一键接管/结束）
RIG_SCRIPT = """
<div id="rig" class="status err" hidden></div>
<div id="camerr" class="status err" hidden></div>
<script>
const STEP = __ACTIVE__;
async function rigTick() {
  try {
    const s = await (await fetch("/api/rig")).json();
    const el = document.getElementById("rig");
    if (s.owner && s.owner !== STEP) {
      el.hidden = false;
      el.innerHTML = "「" + s.owner + "」正在用相机/臂（同一时刻只让一个步骤用）。"
        + " 到那个页面点「结束本步骤」，或 <button onclick='rigTake()'>强制接管</button>";
    } else { el.hidden = true; }
    // 相机出问题就说清楚为什么，别让画面一直空转
    const ce = document.getElementById("camerr");
    if (s.camera_error) {
      ce.hidden = false;
      ce.textContent = "相机： " + s.camera_error;
    } else { ce.hidden = true; }
  } catch (e) {}
}
async function rigPost(url, extra) {
  return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(Object.assign({ owner: STEP }, extra || {})) });
}
async function rigTake() { await rigPost("/rig/take"); rigTick(); }
async function rigRelease() { await rigPost("/rig/release"); rigTick(); }
setInterval(rigTick, 2000);
rigTick();
</script>
"""


def page(title: str, body: str, steps, active: str = "", site: str = "标定台") -> str:
    """把一页内容套进统一外壳（顶部导航按步骤列表生成，底部带硬件占用提示）。"""
    nav = "".join(
        f'<a class="{"on" if cls.name == active else ""}" href="/{cls.name}">{cls.title}</a>'
        for cls in steps
    )
    return (
        f'<!doctype html><html lang="zh"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{site} · {title}</title>{STYLE}</head><body>"
        f'<header><a class="brand" href="/">{site}</a><nav>{nav}</nav></header>'
        f"<main>{body}{RIG_SCRIPT.replace('__ACTIVE__', repr(active))}</main></body></html>"
    )


def card_html(step) -> str:
    """首页卡片。前置条件没满足的步骤不给「进入」入口，把原因写在卡片上。"""
    summary = step.summary()
    lines = "".join(f"{line}\n" for line in summary.get("lines", []))
    blocked = step.guard()
    if blocked:
        return (f'<div class="card"><h3>{step.title} <span class="tag">不可用</span></h3>'
                f'<div class="lines">{lines}</div>'
                f'<p class="kv">{blocked}</p></div>')
    done = summary.get("ok")
    tag = f'<span class="tag{" ok" if done else ""}">{"已完成" if done else "待做"}</span>'
    return (f'<div class="card"><h3>{step.title} {tag}</h3>'
            f'<div class="lines">{lines}</div>'
            f'<p><a class="brand" href="/{step.name}">进入 →</a></p></div>')


def stream_response(frames_fn, fps: float = 25.0) -> Response:
    """把"取当前帧"的回调包成 MJPEG 响应（各页面的画面都走它）。"""
    return Response(mjpeg_stream(frames_fn, fps),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


class _StepRoutes:
    """给某个步骤用的路由注册代理：把 endpoint 自动加上步骤前缀。

    Flask 要求 endpoint 全局唯一，而每个步骤模块里都写着 `@app.get("/<步骤>/xxx")`
    加同名的内部函数（_state / _stream / _record …），不处理就会互相撞车。这里统一
    把 endpoint 变成 `<步骤>.<后缀>`，步骤模块里不用关心。
    """

    def __init__(self, app, step):
        self._app = app
        self._step = step
        self._name = step.name

    def get(self, rule: str, **kwargs):
        return self._route(rule, ["GET"], kwargs)

    def post(self, rule: str, **kwargs):
        return self._route(rule, ["POST"], kwargs)

    def _route(self, rule: str, methods: list, kwargs: dict):
        prefix = f"/{self._name}/"

        def decorator(func):
            suffix = rule[len(prefix):] if rule.startswith(prefix) else rule
            endpoint = kwargs.pop("endpoint", f"{self._name}.{suffix.strip('/') or 'index'}")

            def guarded(*a, **kw):
                err = self._step.guard()
                if err:
                    return jsonify({"ok": False, "error": err}), 409
                return func(*a, **kw)

            guarded.__name__ = func.__name__
            self._app.add_url_rule(rule, endpoint=endpoint, view_func=guarded,
                                   methods=methods, **kwargs)
            return func

        return decorator


class Step:
    """一个标定步骤：一页 + 一组动作 + 一段产物摘要。

    子类要填 `name`（也是 URL 与占用标识）、`title`、`desc`、`needs`；
    实现 `summary()`（首页卡片上显示什么）与 `html()`（本页内容）；
    动作路由在 `register()` 里用 `self.app.get/post(...)` 注册。
    """

    name = ""
    title = ""
    desc = ""
    needs: tuple[str, ...] = ()

    def __init__(self):
        self.app = None          # 路由注册代理（步骤里 @self.app.get(...) 用它）
        self.flask_app = None    # 真 Flask app
        self.rig = None
        self._steps: list = []

    # ---------- 骨架 ----------

    def mount(self, app, rig, steps) -> None:
        """挂到 app 上：本页 + 子类注册的动作。"""
        self.rig, self._steps, self.flask_app = rig, steps, app
        app.add_url_rule(f"/{self.name}", endpoint=f"{self.name}.page",
                         view_func=self._view, methods=["GET"])
        self.app = _StepRoutes(app, self)   # endpoint 自动加步骤前缀，避免同名冲突
        self.register()

    def _view(self):
        site = getattr(self.flask_app, "site_name", "标定台")
        err = self.guard()
        if err:
            body = (self.page_head()
                    + f'<div class="status err">{err}</div>'
                    + '<p><a class="brand" href="/">← 回总览</a></p>')
            return page(self.title, body, self._steps, active=self.name, site=site)
        return page(self.title, self.html(), self._steps, active=self.name, site=site)

    def register(self) -> None:
        """注册本步骤的动作路由（子类实现）。"""

    def guard(self) -> str | None:
        """前置条件没满足时返回一句人话：首页不给入口，页面与接口都进不去。"""
        return None

    # ---------- 给子类用 ----------

    def html(self) -> str:
        raise NotImplementedError

    def summary(self) -> dict:
        """首页卡片：{"lines": [...], "ok": bool}。"""
        return {"lines": ["（这个步骤没有产物摘要）"], "ok": False}

    def acquire(self) -> str | None:
        """占用相机/臂；被别的步骤占着时返回一句话（调用方直接回给页面）。"""
        try:
            self.rig.acquire(self.name)
            return None
        except RuntimeError as e:
            return str(e)

    def page_head(self) -> str:
        """步骤页面的标题区（各页长得一样）。"""
        return f"<h1>{self.title}</h1><div class='desc'>{self.desc}</div>"
