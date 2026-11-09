# Description: 手眼标定（2D 单应）——两个部署共用；产物路径由部署绑定。
#
# 画面**必须已按内参去畸变**（运行时也是先去畸变再检测）：没有内参就先去做内参，
# 这一步会拒绝开始，免得标出一份与运行时不符的矩阵。
# 流程：点画面上的点 → 人拖夹爪尖对准 → 记录（读末端 xy）→ 回起始位姿 → 采够后拟合。
import json
import threading

import cv2
import numpy as np
from flask import jsonify, request

from common.homography import (calibrate_2d_from_correspondences,
                               compute_reprojection_errors, format_report)
from common.ui import Step, stream_response


class HandeyeStep(Step):
    name = "handeye"
    title = "手眼"
    desc = ("在画面上点一个桌面上的显眼点 → 按臂上示教按钮进示教、手拖夹爪尖对准该点 → "
            "点「记录」→ 按示教按钮退出示教 → 点「回起始位姿」。采够 ≥4 点（建议 9~16，分散些）"
            "后点「拟合单应」。画面已按内参去畸变。")
    needs = ("camera", "arm")
    # 子类绑定：这个部署的 paths 模块（HANDEYE_* / HOMOGRAPHY / CAMERA_MATRIX / CALIB_POSE）
    paths = None

    def __init__(self):
        super().__init__()
        self.selected = None
        self.image_points: list = []
        self.robot_points: list = []
        self.report = None
        self.hint = "先点画面上的一个点。"
        self._lock = threading.Lock()
        self._load_points()

    # ---------- 存取 ----------

    def _load_points(self) -> None:
        if self.paths.HANDEYE_IMAGE_POINTS.exists() and self.paths.HANDEYE_ROBOT_POINTS.exists():
            try:
                self.image_points = list(np.load(self.paths.HANDEYE_IMAGE_POINTS))
                self.robot_points = list(np.load(self.paths.HANDEYE_ROBOT_POINTS))
                if len(self.image_points) == len(self.robot_points) and self.image_points:
                    print("已从磁盘恢复 %d 个点" % len(self.image_points), flush=True)
                    self.hint = "从磁盘恢复了 %d 个点，接着采。" % len(self.image_points)
                else:
                    self.image_points, self.robot_points = [], []
            except Exception as e:  # noqa: BLE001
                print("恢复点失败:", e, flush=True)
                self.image_points, self.robot_points = [], []

    def _save_points(self) -> None:
        self.paths.HANDEYE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(self.paths.HANDEYE_IMAGE_POINTS, np.array(self.image_points, dtype=np.float32))
        np.save(self.paths.HANDEYE_ROBOT_POINTS, np.array(self.robot_points, dtype=np.float32))

    def guard(self) -> str | None:
        """没内参就别进来了：手眼用的画面必须先去畸变，否则标出来的矩阵与运行时对不上。"""
        return self._require_intrinsics()

    def _require_intrinsics(self) -> str | None:
        if self.rig.has_intrinsics():
            return None
        return ("还没有相机内参，这一步不能开始——标定用的画面必须先去畸变，"
                "与运行时一致。先去「内参」那页标一次。")

    def _frame(self):
        """取一帧并画上已选点（红）与已记录点（绿）。"""
        frame = self.rig.camera().get_frames()["color"]
        if frame is None:
            return None
        if self.selected is not None:
            cv2.circle(frame, tuple(self.selected), 8, (0, 0, 255), -1)
        for (ix, iy) in self.image_points:
            cv2.circle(frame, (int(ix), int(iy)), 6, (0, 255, 0), -1)
        return frame

    def _move_to_start_pose(self) -> str | None:
        """回到起始/观察位姿；失败返回一句话。"""
        if not self.paths.CALIB_POSE.exists():
            return f"没有 {self.paths.CALIB_POSE}（还没定起始位姿）"
        with open(self.paths.CALIB_POSE, encoding="utf-8") as f:
            target = list(json.load(f)["joints_deg"])
        try:
            arm = self.rig.arm()
            arm.enable_and_take_control()
            arm.move_joints(target)
        except Exception as e:  # noqa: BLE001
            return f"动臂失败：{e}"
        return None

    # ---------- 页面 ----------

    def html(self) -> str:
        return f"""
<div class="row">
  <div class="col"><img class="stream" src="/{self.name}/stream" alt="camera"></div>
  <div class="col">
    <div id="hint" class="status">{self.hint}</div>
    <div class="kv">已采点数：<b id="pairs">0</b>　·　内参：<b id="intr">-</b></div>
    <p>
      <button class="primary" onclick="act('record')">记录（当前选点 + 臂末端 xy）</button>
      <button onclick="act('back')">接管臂 + 回起始位姿</button>
      <button onclick="act('pose')">摆到起始位姿</button>
    </p>
    <p>
      <button class="primary" onclick="act('finish')">拟合单应矩阵</button>
      <button class="danger" onclick="if(confirm('清空已采的点（含磁盘上的）？')) act('reset')">清空已采点</button>
    </p>
    <h2>拟合结果</h2>
    <div class="mono" id="report">{self.report or "-"}</div>
  </div>
</div>
<script>
const img = document.querySelector("img.stream");
img.addEventListener("click", (e) => {{
  const rect = img.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) * img.naturalWidth / rect.width);
  const y = Math.round((e.clientY - rect.top) * img.naturalHeight / rect.height);
  fetch("/{self.name}/click", {{method: "POST", headers: {{"Content-Type": "application/json"}},
                                body: JSON.stringify({{x, y}})}}).then(tick);
}});
async function act(what) {{
  const j = await (await fetch("/{self.name}/" + what, {{method: "POST"}})).json();
  const el = document.getElementById("hint");
  el.textContent = j.hint || j.error || "";
  el.className = "status " + (j.ok ? "ok" : "err");
  tick();
}}
async function tick() {{
  try {{
    const s = await (await fetch("/{self.name}/state")).json();
    document.getElementById("pairs").textContent = s.pairs;
    document.getElementById("intr").textContent = s.has_intrinsics ? "已标" : "未标（先去内参页）";
    document.getElementById("report").textContent = s.report || "-";
  }} catch (e) {{}}
}}
setInterval(tick, 900); tick();
</script>"""

    # ---------- 动作 ----------

    def register(self) -> None:
        app = self.app

        @app.get(f"/{self.name}/stream")
        def _stream():
            return stream_response(lambda: self._frame())

        @app.get(f"/{self.name}/state")
        def _state():
            with self._lock:
                pairs = len(self.image_points)
            return jsonify({"pairs": pairs, "report": self.report,
                            "has_intrinsics": self.rig.has_intrinsics(),
                            "owner": self.rig.owner()})

        @app.post(f"/{self.name}/click")
        def _click():
            err = self.acquire()
            if err:
                return jsonify({"ok": False, "error": err})
            data = request.get_json(silent=True) or {}
            with self._lock:
                self.selected = (int(data.get("x", 0)), int(data.get("y", 0)))
            self.hint = ("已选点 %s。按臂上示教按钮进示教，拖夹爪尖对准它，然后点「记录」。"
                         % (self.selected,))
            return jsonify({"ok": True, "hint": self.hint})

        @app.post(f"/{self.name}/record")
        def _record():
            err = self.acquire() or self._require_intrinsics()
            if err:
                return jsonify({"ok": False, "error": err})
            with self._lock:
                sel = self.selected
            if sel is None:
                return jsonify({"ok": False, "error": "先在画面里点一个点"})
            try:
                pos = self.rig.arm().end_pos()
            except Exception as e:  # noqa: BLE001
                return jsonify({"ok": False, "error": f"读末端位姿失败：{e}"})
            with self._lock:
                self.robot_points.append(np.array(pos[:2], dtype=np.float32))
                self.image_points.append(np.array(sel, dtype=np.float32))
                self.selected = None
                self._save_points()
                n = len(self.image_points)
            self.hint = "已记录 #%d。退出示教后点「接管臂 + 回起始位姿」，再来下一个点。" % n
            print("已记录 #%d 图像点%s -> 末端xy(%.4f, %.4f)" % (n, sel, pos[0], pos[1]), flush=True)
            return jsonify({"ok": True, "hint": self.hint, "pairs": n})

        @app.post(f"/{self.name}/pose")
        def _pose():
            err = self.acquire()
            if err:
                return jsonify({"ok": False, "error": err})
            err = self._move_to_start_pose()
            if err:
                return jsonify({"ok": False, "error": err})
            self.hint = "已回到起始位姿，可以点下一个点。"
            return jsonify({"ok": True, "hint": self.hint})

        @app.post(f"/{self.name}/back")
        def _back():
            """接管臂（人已退出示教）并回到起始位姿——记录完一个点后走这个。"""
            err = self.acquire() or self._require_intrinsics()
            if err:
                return jsonify({"ok": False, "error": err})
            try:
                arm = self.rig.arm()
                if arm.status().teach_status == 0x01:
                    return jsonify({"ok": False,
                                    "error": "仍在示教中——先按臂上示教按钮退出示教，再点这个"})
                if not arm.take_over():
                    return jsonify({"ok": False, "error": "接管失败"})
            except Exception as e:  # noqa: BLE001
                return jsonify({"ok": False, "error": str(e)})
            err = self._move_to_start_pose()
            if err:
                return jsonify({"ok": False, "error": err})
            self.hint = "已回到起始位姿，可以点下一个点。"
            return jsonify({"ok": True, "hint": self.hint})

        @app.post(f"/{self.name}/finish")
        def _finish():
            err = self.acquire() or self._require_intrinsics()
            if err:
                return jsonify({"ok": False, "error": err})
            with self._lock:
                image_points = np.array(self.image_points, dtype=np.float32)
                robot_points = np.array(self.robot_points, dtype=np.float32)
            if len(image_points) < 4:
                return jsonify({"ok": False, "error": f"标定点不足（{len(image_points)} < 4）"})
            try:
                homography, inlier = calibrate_2d_from_correspondences(image_points, robot_points)
                self.paths.HANDEYE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(self.paths.HOMOGRAPHY, homography)
                np.save(self.paths.HANDEYE_IMAGE_POINTS, image_points)
                np.save(self.paths.HANDEYE_ROBOT_POINTS, robot_points)
                errors = compute_reprojection_errors(homography, image_points, robot_points)
                report = format_report(homography, image_points, robot_points, inlier)
            except Exception as e:  # noqa: BLE001
                return jsonify({"ok": False, "error": f"拟合失败：{e}"})
            (self.paths.HANDEYE_DIR / "2d_homography_report.txt").write_text(report, encoding="utf-8")
            with self._lock:
                self.report = report
            print(report, flush=True)
            print("已保存: %s" % self.paths.HOMOGRAPHY, flush=True)
            self.hint = ("完成：平均误差 %.1f mm / 最大 %.1f mm（%d 点）。把矩阵抄进清单的 "
                         "homography_matrix（两处：grasp_pose 与 pick）。"
                         % (errors.mean() * 1000, errors.max() * 1000, len(image_points)))
            return jsonify({"ok": True, "hint": self.hint, "report": report})

        @app.post(f"/{self.name}/reset")
        def _reset():
            # 不抢硬件：这一步只是删文件、清内存，不碰相机和臂。
            # 原来先 acquire 会让"别的步骤开着"时清空按钮失效（看着像坏了）。
            with self._lock:
                self.image_points, self.robot_points, self.selected, self.report = [], [], None, None
            for p in (self.paths.HANDEYE_IMAGE_POINTS, self.paths.HANDEYE_ROBOT_POINTS,
                      self.paths.HOMOGRAPHY):
                if p.exists():
                    p.unlink()
            self.hint = "已清空，可以重新采点。"
            return jsonify({"ok": True, "hint": self.hint})

    # ---------- 摘要 ----------

    def summary(self) -> dict:
        lines = []
        ok = self.paths.HOMOGRAPHY.exists()
        if ok:
            H = np.load(self.paths.HOMOGRAPHY)
            lines.append("已有单应矩阵（artifacts/handeye/2d_homography.npy）")
            lines.append("待抄进清单: " + str(np.round(H, 8).tolist()))
        else:
            lines.append("还没有单应矩阵。")
        with self._lock:
            if self.image_points:
                lines.append(f"当前采了 {len(self.image_points)} 个点")
        return {"ok": ok, "lines": lines}
