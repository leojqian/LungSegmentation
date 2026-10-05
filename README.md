# lungmap

lungmap looks at a chest X-ray scan (a DICOM file, usually ending in `.dcm`)
and, on every frame, finds both lungs and marks four points on each: the
**apex** (top of the lung), the **diaphragm point** (middle of the diaphragm
under the lung) and the two **bottom corners**. You get a spreadsheet of those
points, plus a video and a DICOM of the scan with the points drawn on.

**Everything runs on your own computer. Scans are never uploaded.** The
internet is used only on the first run, to download the program's parts.

**Contents:** [Setup on Windows](#setup-on-windows-one-time-about-10-minutes) ·
[Analyze a scan](#analyze-a-scan) · [Your results](#your-results) ·
[If something goes wrong](#if-something-goes-wrong) · [Mac](#on-a-mac) ·
[Update or remove](#update-or-remove) · [For developers](#for-developers)

---

## Setup on Windows (one time, about 10 minutes)

**You need:** a Windows 10 or 11 PC with an Intel or AMD processor (Windows on
ARM / Snapdragon won't work), about **2 GB** of free disk space, and internet
for the first run.

### 1. Download

1. On this page, click the green **Code** button (top right of the file list).
2. Click **Download ZIP**. `LungSegmentation-main.zip` is saved to your
   **Downloads** folder.

### 2. Unzip it to a permanent place

1. In **File Explorer**, open **Downloads**.
2. Right-click `LungSegmentation-main.zip` and choose **Properties**. If there's
   an **Unblock** box at the bottom, tick it, then click **OK**. This avoids a
   security warning later.
3. Right-click the ZIP again, choose **Extract All...**, pick your
   **Documents** folder with **Browse...**, and click **Extract**.

You now have `Documents\LungSegmentation-main`. It holds many files, but **the
only one you need is `Start lungmap (Windows)`.** Don't use
`Start lungmap (Mac)`; that one is for Apple computers.

### 3. Make a desktop shortcut

Right-click **Start lungmap (Windows)**, then choose **Send to > Desktop
(create shortcut)**. On Windows 11, choose **Show more options** first.

### 4. Run it once to finish setup

1. Double-click the new desktop shortcut.
2. If a blue **"Windows protected your PC"** box appears, click **More info**,
   then **Run anyway**. If an **"Open File - Security Warning"** box appears,
   click **Run**. This only happens the first time.
3. A black window opens and sets everything up. **This takes several minutes**
   (about 1.5 GB to download), and lots of scrolling text is normal.
4. Setup is done when it asks:
   ```
   Drag a DICOM file or a folder into this window, then press Enter:
   ```
   Carry on with the next section.

---

## Analyze a scan

Either:

- **Drag a scan file, or a folder of scans, onto the desktop shortcut.** Or:
- **Double-click the shortcut**, drag the scan or folder into the black window,
  and press **Enter**.

Then wait:

- The very first time, it also downloads the model (about 126 MB).
- A progress bar shows each scan. A scan takes about 1 to 3 minutes.
- When it prints **`Done.`**, the results folder opens by itself. Press any key
  to close the black window.

---

## Your results

Results go in a **`lungmap_output`** folder next to the scan:

```
Your scans folder
├── chest.dcm
└── lungmap_output
    ├── chest_points.csv      the numbers
    ├── chest_overlay.mp4     the video
    └── chest_overlay.dcm     the same pictures, as a DICOM
```

| file | open it with | what it shows |
|---|---|---|
| `chest_points.csv` | Excel | One row per frame with the x, y position (in pixels, from the image's top-left corner) of each lung's apex, diaphragm point and two bottom corners. A blank cell means that point wasn't found on that frame. |
| `chest_overlay.mp4` | Media Player (double-click) | The scan frame by frame, with the lungs and points drawn on. |
| `chest_overlay.dcm` | a DICOM viewer | The same pictures, filed under the same patient and study as the original. |

**Reading the pictures:**

| you see | it means |
|---|---|
| amber lung, **R** in the key | the lung on the image's left (the patient's right lung) |
| blue lung, **L** in the key | the other lung |
| triangle | apex: the top of the lung |
| two squares joined by a dashed line | the bottom corners |
| line along the bottom of the lung | the diaphragm; dashed where its hidden dome is estimated |
| filled circle | the diaphragm point: the middle of that line |
| hollow circle | a simple guess for comparison: the middle of the dashed line between the corners |

After each scan the black window also prints how far apart the diaphragm point
and the simple guess were, on average. That's for comparison only; there's
nothing you need to do with it.

---

## If something goes wrong

| what you see | what to do |
|---|---|
| The black window flashes and closes | Check that you unzipped the folder (step 2) and are using **Start lungmap (Windows)**. |
| "Windows protected your PC" | Click **More info**, then **Run anyway**. |
| Antivirus blocks it | Allow it once. It's a small script that sets up Python. |
| `could not download the model` | Check the internet connection and run it again. |
| `could not load TensorFlow` | Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), restart the computer, and run it again. |
| `not a readable DICOM file` | That file isn't a scan. Check you dragged the right one. |
| `no DICOM files found in: ...` | Drag the folder that directly contains the `.dcm` files. |
| An error about the video or `.mp4` | Move the scans to a folder whose name is only plain English letters and numbers, such as `C:\Scans`. |
| `Something went wrong - see the messages above.` | Take a screenshot of the black window and send it to Leo. |

**Starting over:** type `%LOCALAPPDATA%` into File Explorer's address bar,
press Enter, and delete the **lungmap** folder there. The next run sets
everything up again.

---

## On a Mac

Apple-silicon Macs only. The steps are the same, but use
**Start lungmap (Mac)**. If macOS blocks it the first time, go to **System
Settings > Privacy & Security** and click **Open Anyway**. To give it a scan,
drag the file or folder into the window it opens and press **Return**.

---

## Update or remove

**Update:** download the ZIP again and extract it over the old folder. Your
setup is kept, so it still starts quickly.

**Remove (Windows):**
1. Delete the `LungSegmentation-main` folder and the desktop shortcut.
2. In File Explorer's address bar, open `%LOCALAPPDATA%` and delete the
   **lungmap** and **uv** folders.
3. Open `%APPDATA%` and delete the **uv** folder.
4. Open `%USERPROFILE%\.local\bin` and delete **uv.exe** and **uvx.exe**.

Your scans and `lungmap_output` folders are not touched.

---

## For developers

### Run it from the code

1. **Install uv** (it fetches the right Python by itself):
   - Mac: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. **Get the code:**
   ```bash
   git clone https://github.com/leojqian/LungSegmentation.git
   cd LungSegmentation
   ```
3. **Make a sample** from any chest X-ray DICOM. The first run installs the
   libraries and downloads the model.
   ```bash
   uv run lungmap path/to/scan.dcm -o sample_output --open
   ```

   | option | does |
   |---|---|
   | `path/to/folder` or `"*.dcm"` | process every DICOM in a folder, or every match |
   | `-o DIR` | output folder (default: `lungmap_output` next to each scan) |
   | `--open` | open the output folder when done |
   | `--model PATH` | use a specific weights file instead of the downloaded one |

4. **Run the tests** (about 15 seconds; no model or data needed). Expect
   `238 passed`.
   ```bash
   uv run --extra dev pytest
   ```
5. **Check accuracy against ground truth.** This needs the DDR data in
   `SampleDDR_August2026/` and both model files; see
   [research/README.md](research/README.md). It overwrites
   `outputs/ddr_eval/`.
   ```bash
   uv run --extra research python research/ddr/step07_evaluate.py
   ```
   Expect Dice 0.980 and a diaphragm-point error of 10.66 ± 13.27 mm (right)
   and 37.41 ± 11.37 mm (left).

`pip install .` also works and installs only `lungmap/`, on Python 3.11 to 3.13
(Windows x86-64 or Apple-silicon Mac). The model is taken from `--model`, then
`$LUNGMAP_MODEL`, then `./best_model_ddr_finetuned_phase3.h5`, then a previous
download. If none of those exist, it's downloaded from the `model-v1` release
and its checksum is verified.

### The CSV columns

`frame`, then x and y for each point: `R_apex_x, R_apex_y, R_diaphragm_x,
R_diaphragm_y, R_lower_left_x, ..., L_lower_right_y` (17 columns). `R` is the
image-left lung. Coordinates are in the scan's pixel grid after its stored
left-right flip is applied, the same grid as the overlay files. The diaphragm
point is the horizontal center of the traced diaphragm curve, the DDR
dataset's own definition.

### What's in this repository

| path | what it is | in the package? |
|---|---|---|
| `Start lungmap (Windows).bat`, `Start lungmap (Mac).command` | double-click launchers | runs it |
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

### Limits

Upright, front-facing (PA) chest films only. The model was fine-tuned on 20 DDR
cases, so treat the points as a research measurement, not a clinical one.
