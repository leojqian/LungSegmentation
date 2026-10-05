# lungmap

Finds the lungs on every frame of a chest X-ray DICOM and marks, for each lung,
the **apex**, the **diaphragm point** and the **two bottom corners**. You get a
spreadsheet of those points, plus a video and a DICOM of the scan with
everything drawn on.

## Use it on Windows

1. Click the green **Code** button above, then **Download ZIP**. Unzip it
   somewhere permanent, such as Documents.
2. Drag a DICOM file, or a folder of them, onto **`lungmap.bat`**. Or
   double-click it and drag the scan into the window that opens.
   - If a blue "Windows protected your PC" box appears, click **More info**,
     then **Run anyway**.
3. The first run takes a few minutes, because it downloads about 1.5 GB of
   setup. Later runs start in seconds.
4. When it's done, the `lungmap_output` folder next to your scan opens by
   itself.

## Use it on a Mac (Apple silicon)

Same as Windows, but use **`lungmap.command`**. If macOS blocks it the first
time, go to **System Settings > Privacy & Security** and click **Open Anyway**.

## What you get

| file | contents |
|---|---|
| `<scan>_points.csv` | One row per frame: the 8 points as pixel x, y. Opens in Excel. |
| `<scan>_overlay.mp4` | Video of the scan with lung outlines and points drawn on. |
| `<scan>_overlay.dcm` | The same as a DICOM, filed under the original patient and study. |

**The points.** Each lung has 4 points: `R` is the lung on the image's left (the
patient's right) and `L` is the other.

| point | meaning |
|---|---|
| `apex` | top of the lung |
| `diaphragm` | middle of the traced diaphragm curve (the DDR dataset's definition) |
| `lower_left`, `lower_right` | the lung's bottom corners |

Coordinates are pixels from the top-left of the image (after the scan's stored
left-right flip). A blank cell means that point couldn't be found on that
frame.

**Naive comparison.** The midpoint of each lung's two bottom corners is a
simple guess at the diaphragm point. The overlay draws it as a hollow circle
beside the real one (filled), and the window prints how far apart they are on
average.

## If something goes wrong

| message | fix |
|---|---|
| "could not load TensorFlow" (Windows) | Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), then run again. |
| "could not download the model" | Check the internet connection and run again. |
| Error writing the video (Windows) | Use a folder whose name has only plain English letters. |
| Anything else | Delete `%LOCALAPPDATA%\lungmap` (Windows) or `~/Library/Application Support/lungmap` (Mac), then run again to reinstall. |

## What's in this repository

| path | what it is | in the package? |
|---|---|---|
| `lungmap.bat`, `lungmap.command` | double-click launchers (Windows, Mac) | runs it |
| `lungmap/` | **the lungmap package**: everything the tool runs | **yes** |
| `tests/` | tests for `lungmap/` | no |
| `research/` | the scripts that trained and evaluated the model ([research/README.md](research/README.md)) | no |
| `docs/design/` | design notes behind the algorithm and the fine-tuning | no |

Inside `lungmap/`:

| module | job |
|---|---|
| `cli.py` | the `lungmap` command: inputs, outputs, messages |
| `process.py` | one DICOM → points CSV + overlay DICOM + MP4 |
| `points.py` | the 8 output points, the naive midpoint, the CSV layout |
| `model_file.py` | find, download (first run) and load the U-Net weights |
| `segmentation/pipeline.py` | image → lung masks → measurements |
| `geometry/` | lung mask → diaphragm curve (`diaphragm.py`), apex and corners (`landmarks.py`) |
| `formats/` | read and write DICOM (`dicom_io.py`); write MP4 (`video.py`) |
| `rendering/render.py` | drawing the overlays |

## Run it from the code (for developers)

1. **Install uv** (it fetches the right Python by itself):
   - Mac: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. **Get the code:**
   ```bash
   git clone https://github.com/leojqian/LungSegmentation.git
   cd LungSegmentation
   ```
3. **Make a sample.** Point it at any chest X-ray DICOM. The first run installs
   the libraries and downloads the model (a few minutes).
   ```bash
   uv run lungmap path/to/scan.dcm -o sample_output --open
   ```
   Expect `sample_output/` to open with the three files from [What you get](#what-you-get).
   A 233-frame cine takes about a minute.

   | option | does |
   |---|---|
   | `path/to/folder` or `"*.dcm"` | process every DICOM in a folder, or every match |
   | `-o DIR` | output folder (default: `lungmap_output` next to each scan) |
   | `--open` | open the output folder when done |
   | `--model PATH` | use a specific weights file instead of the downloaded one |

4. **Run the tests** (about 15 seconds; no model or data needed):
   ```bash
   uv run --extra dev pytest
   ```
   Expect `238 passed`.
5. **Check accuracy against ground truth** (needs the DDR data in
   `SampleDDR_August2026/` and both model files; see
   [research/README.md](research/README.md)):
   ```bash
   uv run --extra research python research/ddr/step07_evaluate.py
   ```
   Expect Dice 0.980 and a diaphragm-point error of 10.66 ± 13.27 mm (right)
   and 37.41 ± 11.37 mm (left). This overwrites `outputs/ddr_eval/`.

`pip install .` also works and installs only `lungmap/`, on Python 3.11 to 3.13
(Windows x86-64 or an Apple-silicon Mac). The model is looked up in this
order:
1. `--model`
2. `$LUNGMAP_MODEL`
3. `./best_model_ddr_finetuned_phase3.h5`
4. a previous download

If none is found, it's downloaded from the `model-v1` release with its
checksum verified.

## Limits

Upright, front-facing (PA) chest films only. The model was fine-tuned on 20 DDR
cases, so treat the points as a research measurement, not a clinical one.
