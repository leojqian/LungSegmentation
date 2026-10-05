# Step 8: build the "Diaphragm Point Review" HTML report: every annotated DDR
# frame (119, across 20 cases) with the fine-tuned model's mask and diaphragm
# point drawn over the ground truth, plus step 7's accuracy summary. Uses the
# same lungmap functions as step 7, so both report the same points.
#
# Run from the repo root, after step 7:
#   uv run --extra research python research/ddr/step08_review_report.py
# Writes outputs/ddr_visualize/diaphragm_point_review.html (self-contained, ~8MB;
# open in a browser, step frames with the arrow keys).

import base64
import json
import os

import cv2
import numpy as np
import tensorflow as tf

from lungmap.formats.dicom_io import (load_frames, parse_dm_mode_truth, parse_dm_mode_truth_lines,
                                     parse_lung_area_truth, to_model_input, window_params)
from lungmap.geometry.diaphragm import center_of
from lungmap.rendering.render import SIDE_COLORS as SIDE_COLOR
from lungmap.segmentation.pipeline import measure_combined, segment
from step06_finetune import MASK_OPEN_KERNEL, USE_DICOM_WINDOW_TAGS, find_cases

FINETUNED_MODEL = "best_model_ddr_finetuned_phase3.h5"
EVAL_DIR = "outputs/ddr_eval"
OUT_DIR = "outputs/ddr_visualize"
OUT_HTML = os.path.join(OUT_DIR, "diaphragm_point_review.html")

# Mean out-of-fold DDR Dice printed at the end of step 6's run (each fold scored
# by a model that never saw its patients). Hard-coded because the
# finetune_report.json on disk predates the fold_dice field step 6 now writes.
OUT_OF_FOLD_DICE = 0.9681068500007083

TRUTH_MASK_COLOR = (0, 255, 255)   # yellow
PRED_MASK_COLOR = (0, 255, 0)      # green


def draw_overlay(frame, pred_mask, truth_mask, truth_points, truth_lines, pred_points,
                 spacing_mm, max_dim=760, label_scale=0.75):
    """One frame -> annotated BGR canvas.

    Draws, in layering order: predicted mask contour, truth mask contour,
    the full hand-drawn DM-MODE_truth polyline per side (every annotated
    vertex, thin line + small dots -- this is the actual ground truth, not
    a derived summary), the truth diaphragm point (open circle, dataset's
    center-x convention), the predicted diaphragm point (filled circle, same
    convention), and a connecting line.
    """
    canvas = cv2.cvtColor((np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if pred_mask is not None:
        contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, PRED_MASK_COLOR, 2)
    if truth_mask is not None:
        contours, _ = cv2.findContours(truth_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, TRUTH_MASK_COLOR, 2)

    for side, pts in (truth_lines or {}).items():
        color = SIDE_COLOR[side]
        poly = np.asarray(pts, dtype=np.int32)
        cv2.polylines(canvas, [poly], False, color, 1, cv2.LINE_AA)
        for (x, y) in poly:
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)

    for side, truth_xy in (truth_points or {}).items():
        color = SIDE_COLOR[side]
        truth_pt = (int(round(truth_xy[0])), int(round(truth_xy[1])))
        cv2.circle(canvas, truth_pt, 8, color, 2, cv2.LINE_AA)
        pred_xy = (pred_points or {}).get(side)
        if pred_xy is not None:
            pred_pt = (int(round(pred_xy[0])), int(round(pred_xy[1])))
            cv2.circle(canvas, pred_pt, 6, color, -1, cv2.LINE_AA)
            cv2.line(canvas, pred_pt, truth_pt, color, 1, cv2.LINE_AA)

    f = min(1.0, max_dim / max(canvas.shape[:2]))
    return cv2.resize(canvas, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def build_frames(model):
    """Every DDR case, every annotated frame -> list of {"case", "frame",
    "hasMask", "hasPoints", "errors", "img"} dicts for the report."""
    cases = find_cases()
    print(f"{len(cases)} cases")

    out_frames = []
    for case_id, dcm_path, xml_path, raw_path in cases:
        frames, spacing_mm, photometric = load_frames(dcm_path)
        lung_truth = parse_lung_area_truth(raw_path, shape=frames.shape[1:])
        dm_lines = parse_dm_mode_truth_lines(xml_path)   # full polylines, for drawing
        dm_points = parse_dm_mode_truth(xml_path)         # dataset's own center-x point, for scoring
        wc, ww = window_params(dcm_path) if USE_DICOM_WINDOW_TAGS else (None, None)

        for frame_idx in sorted(set(lung_truth) | set(dm_points)):
            img = to_model_input(frames[frame_idx], photometric, wc, ww)
            pred_mask = segment(img, model, open_kernel=MASK_OPEN_KERNEL)
            _, _, pred_curves = measure_combined(pred_mask)
            pred_points = {side: center_of(*curve) for side, curve in pred_curves.items()
                          if curve is not None}

            truth_mask = lung_truth.get(frame_idx)
            truth_points = dm_points.get(frame_idx, {})
            truth_lines = dm_lines.get(frame_idx, {})

            canvas = draw_overlay(img, pred_mask, truth_mask, truth_points, truth_lines,
                                  pred_points, spacing_mm=spacing_mm, max_dim=760,
                                  label_scale=0.75)
            ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 74])
            b64 = base64.b64encode(buf).decode("ascii")

            errors = {}
            for side, truth_xy in truth_points.items():
                pred_xy = pred_points.get(side)
                if pred_xy is not None:
                    err_px = float(np.hypot(pred_xy[0] - truth_xy[0], pred_xy[1] - truth_xy[1]))
                    errors[side] = {"px": round(err_px, 1), "mm": round(err_px * spacing_mm, 1)}

            out_frames.append({
                "case": case_id.replace("_PA_deep", ""),
                "frame": frame_idx,
                "hasMask": truth_mask is not None,
                "hasPoints": bool(truth_points),
                "errors": errors,
                "img": "data:image/jpeg;base64," + b64,
            })
        print(case_id, "done")

    return out_frames


def load_eval_summary(label):
    with open(os.path.join(EVAL_DIR, f"{label}.json")) as f:
        return json.load(f)["summary"]


def main():
    model = tf.keras.models.load_model(FINETUNED_MODEL)
    out_frames = build_frames(model)

    total_kb = sum(len(f["img"]) for f in out_frames) / 1024
    print(f"TOTAL base64 payload: {total_kb:.0f} KB, {len(out_frames)} frames")

    summary = {"baseline": load_eval_summary("baseline"),
              "finetuned": load_eval_summary("finetuned"),
              "outOfFoldDice": OUT_OF_FOLD_DICE}
    payload = json.dumps({"frames": out_frames, "summary": summary})

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(TEMPLATE.replace("__FRAME_DATA__", payload))
    print(f"wrote {OUT_HTML}")


# --- report page (viewer UI) --------------------------------------------------
# Static HTML/CSS/JS shell; __FRAME_DATA__ is replaced with the JSON payload
# built above. No model or metric logic here -- purely presentation.
TEMPLATE = """<meta charset="utf-8">
<title>Diaphragm Point Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f4f6f7;
    --surface: #ffffff;
    --surface-2: #eceff1;
    --border: #d7dde1;
    --text: #16202a;
    --text-dim: #5b6b78;
    --accent: #0e9488;
    --accent-soft: #0e948822;
    --truth: #b8930a;
    --pred: #1d9a4e;
    --side-r: #c9760f;
    --side-l: #2166c9;
    --danger: #c9432f;
    --radius: 10px;
    --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: "IBM Plex Mono", "SF Mono", Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0a0d10;
      --surface: #12161b;
      --surface-2: #1a1f26;
      --border: #262d35;
      --text: #e7ebef;
      --text-dim: #8996a3;
      --accent: #45d8c8;
      --accent-soft: #45d8c81f;
      --truth: #f0cd4a;
      --pred: #4ade80;
      --side-r: #f0a860;
      --side-l: #6fb8f5;
      --danger: #f2665a;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0a0d10;
    --surface: #12161b;
    --surface-2: #1a1f26;
    --border: #262d35;
    --text: #e7ebef;
    --text-dim: #8996a3;
    --accent: #45d8c8;
    --accent-soft: #45d8c81f;
    --truth: #f0cd4a;
    --pred: #4ade80;
    --side-r: #f0a860;
    --side-l: #6fb8f5;
    --danger: #f2665a;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
  }
  ::selection { background: var(--accent-soft); }

  .app {
    display: grid;
    grid-template-columns: 300px 1fr;
    height: 100vh;
    min-height: 560px;
  }
  @media (max-width: 860px) {
    .app { grid-template-columns: 1fr; grid-template-rows: auto 1fr; height: auto; }
    .sidebar { max-height: 40vh; }
  }

  /* ---------- sidebar ---------- */
  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 20px 18px 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .brand {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .brand .kicker {
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--accent);
  }
  .brand h1 {
    margin: 0;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.01em;
    text-wrap: balance;
  }
  .brand .sub {
    color: var(--text-dim);
    font-size: 12.5px;
    line-height: 1.5;
  }

  .panel {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface-2);
    padding: 14px;
  }
  .panel h2 {
    margin: 0 0 10px;
    font-size: 10.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }

  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 10px;
  }
  .stat .label { font-size: 10.5px; color: var(--text-dim); }
  .stat .value {
    font-family: var(--mono);
    font-size: 17px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
  }
  .stat .value small { font-size: 11px; color: var(--text-dim); font-weight: 400; }

  .bar-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 11.5px; }
  .bar-row .side-tag {
    width: 14px; font-family: var(--mono); font-weight: 600; text-align: center;
  }
  .bar-track {
    flex: 1; height: 7px; background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; overflow: hidden;
  }
  .bar-fill { height: 100%; border-radius: 4px 0 0 4px; }
  .bar-row .side-r .bar-fill { background: var(--side-r); }
  .bar-row .side-l .bar-fill { background: var(--side-l); }
  .bar-row .num { font-family: var(--mono); font-variant-numeric: tabular-nums; width: 92px; text-align: right; color: var(--text-dim); }

  .legend { display: flex; flex-direction: column; gap: 7px; font-size: 12px; }
  .legend .row { display: flex; align-items: center; gap: 8px; }
  .legend .swatch { width: 14px; height: 14px; border-radius: 4px; flex-shrink: 0; }
  .legend .swatch.line { border-radius: 0; height: 2px; align-self: center; }
  .legend .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .legend .dot.hollow { background: transparent; border: 2px solid; }

  .filters { display: flex; flex-direction: column; gap: 10px; }
  .seg {
    display: flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  }
  .seg button {
    flex: 1; padding: 7px 4px; border: none; background: var(--surface); color: var(--text-dim);
    font-family: var(--sans); font-size: 11.5px; font-weight: 500; cursor: pointer;
  }
  .seg button + button { border-left: 1px solid var(--border); }
  .seg button.active { background: var(--accent); color: #06231f; font-weight: 600; }
  select {
    width: 100%; padding: 7px 8px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); font-family: var(--mono); font-size: 12px;
  }

  .kbd-hint {
    font-size: 11px; color: var(--text-dim); line-height: 1.6;
  }
  .kbd-hint kbd {
    font-family: var(--mono); background: var(--surface); border: 1px solid var(--border);
    border-bottom-width: 2px; border-radius: 4px; padding: 1px 5px; font-size: 10.5px;
  }

  /* ---------- main stage ---------- */
  .stage {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .stage-top {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; border-bottom: 1px solid var(--border);
    gap: 12px; flex-wrap: wrap;
  }
  .frame-id {
    font-family: var(--mono); font-size: 13px; display: flex; gap: 10px; align-items: baseline;
  }
  .frame-id .case { font-weight: 600; }
  .frame-id .idx { color: var(--text-dim); }
  .chip-row { display: flex; gap: 6px; }
  .chip {
    font-family: var(--mono); font-size: 10.5px; padding: 2px 7px; border-radius: 20px;
    border: 1px solid var(--border); color: var(--text-dim);
  }
  .chip.on.mask { color: var(--truth); border-color: var(--truth); }
  .chip.on.points { color: var(--side-l); border-color: var(--side-l); }

  .error-readout { display: flex; gap: 14px; font-family: var(--mono); font-size: 12px; }
  .error-readout .e { display: flex; gap: 5px; align-items: center; }
  .error-readout .sw { width: 8px; height: 8px; border-radius: 50%; }

  .viewer {
    flex: 1; display: flex; align-items: center; justify-content: center;
    position: relative; padding: 18px; min-height: 0;
    background:
      radial-gradient(ellipse at center, var(--surface-2) 0%, var(--bg) 78%);
  }
  .viewer img {
    max-width: 100%; max-height: 100%; object-fit: contain;
    border-radius: 6px;
    box-shadow: 0 8px 30px -8px rgb(0 0 0 / 0.4);
  }
  .nav-btn {
    position: absolute; top: 50%; transform: translateY(-50%);
    width: 42px; height: 42px; border-radius: 50%;
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    font-size: 17px; cursor: pointer; display: flex; align-items: center; justify-content: center;
    opacity: 0.85;
  }
  .nav-btn:hover { opacity: 1; background: var(--surface-2); }
  .nav-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .nav-btn.prev { left: 14px; }
  .nav-btn.next { right: 14px; }
  .nav-btn:disabled { opacity: 0.25; cursor: default; }

  .filmstrip-wrap {
    border-top: 1px solid var(--border);
    padding: 10px 14px;
    background: var(--surface);
  }
  .filmstrip {
    display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px;
    scroll-behavior: smooth;
  }
  .filmstrip::-webkit-scrollbar { height: 6px; }
  .filmstrip::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  .thumb {
    flex-shrink: 0; width: 54px; height: 54px; border-radius: 6px; overflow: hidden;
    border: 2px solid transparent; cursor: pointer; padding: 0; background: var(--surface-2);
    position: relative;
  }
  .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .thumb.active { border-color: var(--accent); }
  .thumb .tick {
    position: absolute; bottom: 2px; right: 2px; width: 6px; height: 6px; border-radius: 50%;
  }
  .thumb .tick.mask { background: var(--truth); }
  .thumb:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }

  .empty-msg {
    color: var(--text-dim); font-family: var(--mono); font-size: 13px; padding: 40px;
    text-align: center;
  }

  @media (prefers-reduced-motion: reduce) {
    .filmstrip { scroll-behavior: auto; }
  }
</style>

<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <div class="kicker">DDR &middot; diaphragm tracking</div>
      <h1>Diaphragm Point Review</h1>
      <div class="sub">Fine-tuned model's predictions vs. ground-truth lung masks &amp; diaphragm points, every annotated frame, 20 cases. Diaphragm point uses the dataset's own definition (center x-coordinate of the curve, not its peak). Orientation-corrected (DICOM Field of View Horizontal Flip applied), retrained on the corrected data.</div>
    </div>

    <div class="panel" style="border-color:var(--accent)">
      <h2 style="color:var(--accent)">Baseline vs. fine-tuned</h2>
      <table style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px">
        <tr style="color:var(--text-dim)">
          <td></td><td style="text-align:right;padding:2px 6px">baseline</td>
          <td style="text-align:right;padding:2px 6px;color:var(--accent)">fine-tuned</td>
        </tr>
        <tr><td>Dice mean</td>
          <td style="text-align:right;padding:2px 6px" id="cmpDiceBase">&mdash;</td>
          <td style="text-align:right;padding:2px 6px;color:var(--accent);font-weight:600" id="cmpDiceFt">&mdash;</td></tr>
        <tr><td>Dice median</td>
          <td style="text-align:right;padding:2px 6px" id="cmpDiceMedBase">&mdash;</td>
          <td style="text-align:right;padding:2px 6px;color:var(--accent);font-weight:600" id="cmpDiceMedFt">&mdash;</td></tr>
        <tr><td>Point err mean</td>
          <td style="text-align:right;padding:2px 6px" id="cmpApexBase">&mdash;</td>
          <td style="text-align:right;padding:2px 6px;color:var(--accent);font-weight:600" id="cmpApexFt">&mdash;</td></tr>
        <tr><td>Point err median</td>
          <td style="text-align:right;padding:2px 6px" id="cmpApexMedBase">&mdash;</td>
          <td style="text-align:right;padding:2px 6px;color:var(--accent);font-weight:600" id="cmpApexMedFt">&mdash;</td></tr>
        <tr><td>Point err std</td>
          <td style="text-align:right;padding:2px 6px" id="cmpApexStdBase">&mdash;</td>
          <td style="text-align:right;padding:2px 6px;color:var(--accent);font-weight:600" id="cmpApexStdFt">&mdash;</td></tr>
      </table>
      <div class="sub" style="margin:10px 0 0">Mean out-of-fold DDR Dice (5-fold cross-val, stricter estimate): <strong id="cmpOutOfFold">&mdash;</strong></div>
    </div>

    <div class="panel">
      <h2>Fine-tuned point error, by side</h2>
      <div class="bar-row side-r">
        <div class="side-tag" style="color:var(--side-r)">R</div>
        <div class="bar-track"><div class="bar-fill" id="barR"></div></div>
        <div class="num" id="numR">&mdash;</div>
      </div>
      <div class="bar-row side-l">
        <div class="side-tag" style="color:var(--side-l)">L</div>
        <div class="bar-track"><div class="bar-fill" id="barL"></div></div>
        <div class="num" id="numL">&mdash;</div>
      </div>
    </div>

    <div class="panel">
      <h2>Legend</h2>
      <div class="legend">
        <div class="row"><span class="swatch line" style="background:var(--truth)"></span> Ground-truth mask outline</div>
        <div class="row"><span class="swatch line" style="background:var(--pred)"></span> Predicted mask outline</div>
        <div class="row"><span class="dot hollow" style="border-color:var(--side-r)"></span> Truth diaphragm point (R / L)</div>
        <div class="row"><span class="dot" style="background:var(--side-r)"></span> Predicted diaphragm point (R / L)</div>
      </div>
    </div>

    <div class="panel filters">
      <h2>Show</h2>
      <div class="seg" id="filterSeg">
        <button data-filter="all" class="active">All 119</button>
        <button data-filter="mask">Mask &middot; 40</button>
        <button data-filter="points">Points only</button>
      </div>
      <select id="caseSelect"></select>
    </div>

    <div class="kbd-hint">
      <kbd>&larr;</kbd> <kbd>&rarr;</kbd> step frames &middot; click filmstrip to jump<br>
      works from anywhere on this page &mdash; no need to click the image first
    </div>
  </aside>

  <main class="stage">
    <div class="stage-top">
      <div class="frame-id">
        <span class="case" id="hdrCase">&mdash;</span>
        <span class="idx" id="hdrFrame">&mdash;</span>
        <span class="idx" id="hdrCount">&mdash;</span>
      </div>
      <div class="chip-row" id="hdrChips"></div>
      <div class="error-readout" id="hdrErrors"></div>
    </div>

    <div class="viewer">
      <button class="nav-btn prev" id="btnPrev" aria-label="Previous frame">&#8592;</button>
      <img id="stageImg" alt="Predicted vs truth overlay">
      <button class="nav-btn next" id="btnNext" aria-label="Next frame">&#8594;</button>
    </div>

    <div class="filmstrip-wrap">
      <div class="filmstrip" id="filmstrip"></div>
    </div>
  </main>
</div>

<script>
const DATA = __FRAME_DATA__;

const state = { filter: "all", idx: 0, view: [] };

function applyFilter() {
  const f = state.filter;
  state.view = DATA.frames
    .map((fr, i) => ({ ...fr, _i: i }))
    .filter(fr => f === "all" ? true : f === "mask" ? fr.hasMask : !fr.hasMask);
  if (state.idx >= state.view.length) state.idx = 0;
}

function fmtMm(v) { return v == null ? "&mdash;" : v.toFixed(1) + "mm"; }

function renderStats() {
  const base = DATA.summary.baseline, ft = DATA.summary.finetuned;
  document.getElementById("cmpDiceBase").textContent = base.dice_mean.toFixed(3);
  document.getElementById("cmpDiceFt").textContent = ft.dice_mean.toFixed(3);
  document.getElementById("cmpDiceMedBase").textContent = base.dice_median.toFixed(3);
  document.getElementById("cmpDiceMedFt").textContent = ft.dice_median.toFixed(3);
  document.getElementById("cmpApexBase").textContent = base.apex_mm_mean.toFixed(1) + "mm";
  document.getElementById("cmpApexFt").textContent = ft.apex_mm_mean.toFixed(1) + "mm";
  document.getElementById("cmpApexMedBase").textContent = base.apex_mm_median.toFixed(1) + "mm";
  document.getElementById("cmpApexMedFt").textContent = ft.apex_mm_median.toFixed(1) + "mm";
  document.getElementById("cmpApexStdBase").textContent = base.apex_mm_std.toFixed(1) + "mm";
  document.getElementById("cmpApexStdFt").textContent = ft.apex_mm_std.toFixed(1) + "mm";
  document.getElementById("cmpOutOfFold").textContent = DATA.summary.outOfFoldDice.toFixed(3);

  const maxMm = Math.max(ft.apex_mm_mean_R, ft.apex_mm_mean_L);
  document.getElementById("barR").style.width = (100 * ft.apex_mm_mean_R / maxMm) + "%";
  document.getElementById("barL").style.width = (100 * ft.apex_mm_mean_L / maxMm) + "%";
  document.getElementById("numR").textContent = ft.apex_mm_mean_R.toFixed(1) + " ± " + ft.apex_mm_std_R.toFixed(1) + "mm";
  document.getElementById("numL").textContent = ft.apex_mm_mean_L.toFixed(1) + " ± " + ft.apex_mm_std_L.toFixed(1) + "mm";
}

function renderCaseSelect() {
  const cases = [...new Set(DATA.frames.map(f => f.case))];
  const sel = document.getElementById("caseSelect");
  sel.innerHTML = '<option value="">Jump to case&hellip;</option>' +
    cases.map(c => `<option value="${c}">${c}</option>`).join("");
  sel.onchange = () => {
    if (!sel.value) return;
    const i = state.view.findIndex(f => f.case === sel.value);
    if (i >= 0) { state.idx = i; render(); }
    sel.value = "";
  };
}

function renderFilmstrip() {
  const strip = document.getElementById("filmstrip");
  strip.innerHTML = "";
  state.view.forEach((fr, i) => {
    const b = document.createElement("button");
    b.className = "thumb" + (i === state.idx ? " active" : "");
    b.innerHTML = `<img src="${fr.img}" loading="lazy" alt="">` +
      (fr.hasMask ? '<span class="tick mask"></span>' : "");
    b.title = `${fr.case} f${fr.frame}`;
    b.onclick = () => { state.idx = i; render(); };
    strip.appendChild(b);
  });
}

function render() {
  const fr = state.view[state.idx];
  const img = document.getElementById("stageImg");
  const empty = state.view.length === 0;

  document.getElementById("btnPrev").disabled = empty || state.idx === 0;
  document.getElementById("btnNext").disabled = empty || state.idx === state.view.length - 1;

  if (empty) {
    img.style.display = "none";
    document.getElementById("hdrCase").textContent = "No frames";
    document.getElementById("hdrFrame").textContent = "";
    document.getElementById("hdrCount").textContent = "";
    document.getElementById("hdrChips").innerHTML = "";
    document.getElementById("hdrErrors").innerHTML = "";
    return;
  }
  img.style.display = "";
  img.src = fr.img;
  document.getElementById("hdrCase").textContent = fr.case;
  document.getElementById("hdrFrame").textContent = "frame " + fr.frame;
  document.getElementById("hdrCount").textContent = (state.idx + 1) + " / " + state.view.length;

  document.getElementById("hdrChips").innerHTML =
    `<span class="chip ${fr.hasMask ? "on mask" : ""}">mask</span>` +
    `<span class="chip ${fr.hasPoints ? "on points" : ""}">points</span>`;

  const errs = fr.errors || {};
  document.getElementById("hdrErrors").innerHTML = Object.entries(errs).map(([side, e]) =>
    `<span class="e"><span class="sw" style="background:var(--side-${side.toLowerCase()})"></span>${side} ${e.px}px / ${e.mm}mm</span>`
  ).join("");

  // keep filmstrip active state + scroll position in sync without a full rebuild
  document.querySelectorAll(".thumb").forEach((t, i) => {
    t.classList.toggle("active", i === state.idx);
    if (i === state.idx) t.scrollIntoView({ block: "nearest", inline: "center" });
  });
}

function setFilter(f) {
  state.filter = f;
  state.idx = 0;
  document.querySelectorAll("#filterSeg button").forEach(b =>
    b.classList.toggle("active", b.dataset.filter === f));
  applyFilter();
  renderFilmstrip();
  render();
}

document.getElementById("filterSeg").addEventListener("click", e => {
  const btn = e.target.closest("button[data-filter]");
  if (btn) setFilter(btn.dataset.filter);
});

document.getElementById("btnPrev").onclick = () => { if (state.idx > 0) { state.idx--; render(); } };
document.getElementById("btnNext").onclick = () => { if (state.idx < state.view.length - 1) { state.idx++; render(); } };

window.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { if (state.idx > 0) { state.idx--; render(); } e.preventDefault(); }
  if (e.key === "ArrowRight") { if (state.idx < state.view.length - 1) { state.idx++; render(); } e.preventDefault(); }
});

renderStats();
applyFilter();
renderCaseSelect();
renderFilmstrip();
render();
</script>
"""


if __name__ == "__main__":
    main()
