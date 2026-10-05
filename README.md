# lungmap

Finds the lungs on every frame of a chest X-ray DICOM and marks, for each lung,
the apex, the diaphragm point and the two bottom corners. You get a spreadsheet
of those points plus a video and a DICOM of the scan with everything drawn on.

## Quick start (Windows)

1. On this page, click the green **Code** button, then **Download ZIP**.
2. Open the downloaded ZIP, and drag the folder inside it somewhere permanent,
   such as your Documents folder.
3. In that folder, double-click **`lungmap.bat`**. Or drag a DICOM file, or a
   whole folder of them, onto `lungmap.bat`.
   - If a blue "Windows protected your PC" box appears, click **More info**,
     then **Run anyway**.
   - If you double-clicked, a black window asks for a scan. Drag a DICOM file
     or folder into that window and press **Enter**.
4. **The first run takes several minutes.** It downloads Python, the libraries
   and the model, about 1.5 GB in all. Later runs start in seconds.
5. When it finishes, the results folder opens by itself. It is called
   `lungmap_output` and sits next to your scan. Press any key to close the
   black window.

## Quick start (Mac, Apple silicon)

Same as Windows, but double-click **`lungmap.command`** instead.

- The very first time, macOS may refuse to open it. Open **System Settings >
  Privacy & Security**, scroll down, and click **Open Anyway**.
- To give it a scan, drag the file or folder into the window it opens and
  press **Return**.

## What you get

For each scan `<name>.dcm`, the `lungmap_output` folder gets three files:

| file | what it is |
|---|---|
| `<name>_points.csv` | Opens in Excel. One row per frame, with the 8 points as pixel coordinates. |
| `<name>_overlay.mp4` | A video of the scan with lung outlines and points drawn on. |
| `<name>_overlay.dcm` | The same as a DICOM, for a DICOM viewer. It is filed under the same patient and study as the original. |

### The 8 points

For each lung (`R` is the lung on the image's left, which is the patient's
right on a front-facing film; `L` is the other one):

| point | meaning |
|---|---|
| `apex` | the top of the lung |
| `diaphragm` | the diaphragm point: the horizontal middle of the traced diaphragm curve (the DDR dataset's own definition) |
| `lower_left`, `lower_right` | the lung's two bottom corners (the costophrenic and cardiophrenic angles) |

The CSV columns are `frame`, then `R_apex_x, R_apex_y, R_diaphragm_x, ...,
L_lower_right_y` (17 in all). Coordinates are pixels, counted from the top-left
of the image after its stored left-right flip is applied. The overlay files use
the same pixel grid. A point that couldn't be measured on a frame is left blank.

### Naive baseline

The midpoint of each lung's two bottom corners is a naive guess at the
diaphragm point, with no curve tracing or fitting. The overlay shows it as a
hollow circle next to the real diaphragm point (filled). At the end of each
scan, the window prints how far apart the two are on average, per side, in
pixels and mm.

## Troubleshooting

- **"could not load TensorFlow" on Windows:** install the [Microsoft Visual C++
  Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), then run
  again.
- **"could not download the model":** check the internet connection and run
  again. The download starts over, and a broken download is never kept.
- **Output folder names on Windows** must be plain English letters. OpenCV
  can't write the video into folders with accented or non-Latin characters.
- **Starting over:** delete `%LOCALAPPDATA%\lungmap` on Windows, or
  `~/Library/Application Support/lungmap` on Mac. The next run reinstalls.

## Command line

The launchers run the `lungmap` command, which you can also use directly:

```bash
pip install .                      # or: uv tool install .
lungmap scan.dcm                   # files, folders, or wildcards like *.dcm
lungmap scans/ -o results --open   # choose the output folder; open it when done
```

`lungmap` finds the model in this order:
1. `--model PATH`
2. the `LUNGMAP_MODEL` environment variable
3. `best_model_ddr_finetuned_phase3.h5` in the current folder
4. a copy downloaded on an earlier run

If none of those exist, it downloads the model (~126MB, checksum-verified) from
this repo's `model-v1` release into a per-user folder. Python 3.11 to 3.13 is
required, on Windows x86-64 or Apple-silicon Macs. TensorFlow 2.21 has no build
for Intel Macs.

## Limits

- Upright, front-facing (PA) chest films only. The diaphragm tracing assumes
  that view.
- The model was fine-tuned on 20 DDR cases. Treat the points as a research
  measurement, not a clinical one.

## Development

```bash
uv run --extra dev pytest          # or: pip install -e ".[dev]" && pytest
```

The library lives in `lungmap/` (`geometry/`, `formats/`, `segmentation/`,
`rendering/`, plus `points.py`, `process.py`, `model.py` and `cli.py`). The
`Step*.py` and `submission/` scripts are the research and training pipeline that
produced the model; run them from the repo root.
