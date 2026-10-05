# lungmap on Windows: getting started

lungmap looks at a chest X-ray scan (a DICOM file, usually ending in `.dcm`)
and, on every frame, finds both lungs and marks four points on each:

- the **apex** (the top of the lung)
- the **diaphragm point** (the middle of the diaphragm under the lung)
- the two **bottom corners** of the lung

You get a spreadsheet of those points, plus a video and a DICOM of the scan
with the points drawn on.

**Everything runs on your own computer. Your scans are never uploaded.** The
internet is used only the first time, to download the program's parts.

---

## What you need

- A Windows 10 or 11 PC with a normal Intel or AMD processor. Windows on ARM
  (Snapdragon) machines won't work.
- About **2 GB of free disk space**.
- An internet connection for the first run.
- Your scans as DICOM files.

---

## Part 1: One-time setup (about 10 minutes)

### Step 1: Download

1. Open this page in your web browser:
   **https://github.com/leojqian/LungSegmentation**
2. Click the green **Code** button near the top right.
3. Click **Download ZIP**. A file called `LungSegmentation-main.zip` is saved to
   your **Downloads** folder.

### Step 2: Unzip it to a permanent place

1. Open **File Explorer** and go to **Downloads**.
2. *Optional, but it avoids a security warning later:* right-click
   `LungSegmentation-main.zip`, choose **Properties**, tick **Unblock** near the
   bottom (if it's there), then click **OK**.
3. Right-click `LungSegmentation-main.zip` and choose **Extract All...**
4. Choose your **Documents** folder (click **Browse...**), then click
   **Extract**.

You now have a folder `Documents\LungSegmentation-main`. It contains many
files. **You only ever need one: `Start lungmap (Windows)`.** Ignore the rest.

> Don't run it from inside the ZIP, and don't use the file called
> `Start lungmap (Mac)`. That one is for Apple computers.

### Step 3: Make a desktop shortcut (recommended)

1. In the `LungSegmentation-main` folder, right-click
   **Start lungmap (Windows)**.
2. Windows 11: choose **Show more options**, then **Send to > Desktop (create
   shortcut)**. Windows 10: choose **Send to > Desktop (create shortcut)**.

A **Start lungmap (Windows) - Shortcut** icon appears on your desktop. Use it
from now on.

### Step 4: The first run

1. Double-click the shortcut.
2. If a blue box says **"Windows protected your PC"**, click **More info**, then
   **Run anyway**. If a box says **"Open File - Security Warning"**, click
   **Run**. This only happens the first time.
3. A black window opens and starts setting up. **This takes several minutes**
   and downloads about 1.5 GB. Lots of text scrolling past is normal. You'll
   see lines such as:
   ```
   First run: installing uv, which sets up Python for lungmap ...
   Downloading cpython-3.11 ...
   Downloading tensorflow ...
   ```
4. When it's ready it asks:
   ```
   Drag a DICOM file or a folder into this window, then press Enter:
   >
   ```
   Go on to Part 2.

Later runs skip all of this and start in a few seconds.

---

## Part 2: Analyzing a scan

### Way A: drag and drop (easiest)

Drag a scan file, or a whole folder of scans, from File Explorer and **drop it
onto the desktop shortcut**.

### Way B: from the window

1. Double-click the shortcut.
2. When it asks, drag the scan file or folder **into the black window**. Its
   location appears as text.
3. Press **Enter**.

### Then

- The first time, it also downloads the model (about 126 MB):
  `First run: downloading the model (~126MB) ...`
- A progress bar shows each scan: `45%|████      | 105/233 [00:40<00:50]`. A
  scan takes about 1 to 3 minutes, depending on the computer.
- When it finishes, it prints **`Done.`** and the results folder opens by
  itself.
- Press any key to close the black window.

---

## Part 3: Your results

The results go in a folder called **`lungmap_output`**, created **next to your
scan**. For a scan called `chest.dcm`:

```
Your scans folder
├── chest.dcm
└── lungmap_output
    ├── chest_points.csv      <- the numbers
    ├── chest_overlay.mp4     <- the video
    └── chest_overlay.dcm     <- the same, as a DICOM
```

| file | open it with | what it shows |
|---|---|---|
| `chest_points.csv` | Excel (double-click) | One row per frame: the x, y position of each point, in pixels from the image's top-left corner. An empty cell means that point couldn't be found on that frame. |
| `chest_overlay.mp4` | Double-click (Media Player) | The scan playing frame by frame, with the lungs and points drawn on. |
| `chest_overlay.dcm` | Your DICOM viewer | The same pictures, stored under the same patient and study as the original. |

### Reading the pictures

- **Colors:** the lung on the **left of the image** (the patient's right lung,
  called **R**) is tinted **amber**. The other lung (**L**) is **blue**. The key
  in the top-left corner shows this, plus the frame number.
- **Apex:** the **triangle** at the top of each lung.
- **Bottom corners:** the two **squares**, with a dashed line between them.
- **Diaphragm curve:** the **line along the bottom of the lung**. Where it turns
  dashed, the program is estimating the dome's shape where it's hidden.
- **Diaphragm point:** the **filled circle**, the middle of that curve.
- **Naive guess:** the **hollow circle**, the middle of the dashed line between
  the corners. It's a simple guess, shown for comparison.

### The numbers in the black window

After each scan the window prints something like:

```
chest.dcm: 233 frames (0 with a point missing)
  naive corner midpoint vs diaphragm point (pixel spacing 0.400 mm):
    R: mean 40.5 px (16.2 mm), median 33.2 px (13.3 mm), n=233
    L: mean 13.8 px (5.5 mm), median 5.5 px (2.2 mm), n=233
```

This is how far apart, on average, the diaphragm point and the simple guess
were for each lung. It's for comparison only. You don't need to do anything
with it.

---

## If something goes wrong

| what you see | what to do |
|---|---|
| The black window flashes and disappears | Make sure you unzipped the folder (Step 2) and are using **Start lungmap (Windows)**, not the Mac one. |
| A blue "Windows protected your PC" box | Click **More info**, then **Run anyway**. |
| Your antivirus blocks it | Allow it once. It's a small script that sets up Python. |
| `could not download the model` | Check the internet connection and run it again. A half-finished download is never kept. |
| `could not load TensorFlow` | Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) (run the downloaded file, click Install), restart the computer, and run it again. |
| `not a readable DICOM file` | That file isn't a DICOM scan. Check you dragged the right file. |
| `no DICOM files found in: ...` | The folder you dragged has no scans directly inside it. Drag the folder that holds the `.dcm` files. |
| An error mentioning the video, `.mp4` or `could not open` | Move the scans into a folder whose name uses only plain English letters and numbers, such as `C:\Scans`, and try again. |
| `Something went wrong - see the messages above.` | Take a photo or screenshot of the black window and send it to Leo. |

**Starting over:** if setup was interrupted or something seems broken, type
`%LOCALAPPDATA%` into File Explorer's address bar, press Enter, and delete the
**lungmap** folder there. The next run sets everything up again.

---

## Getting a newer version

When Leo tells you there's an update:

1. Download the ZIP again (Step 1) and extract it to the same place,
   replacing the old folder.
2. Run it as usual. Your setup is kept, so it starts quickly (it may download
   a few updated parts).

Your desktop shortcut keeps working as long as the folder stays in the same
place with the same name.

## Removing it completely

1. Delete the `LungSegmentation-main` folder and the desktop shortcut.
2. Type `%LOCALAPPDATA%` into File Explorer's address bar and delete the
   **lungmap** and **uv** folders there.
3. Type `%APPDATA%` into the address bar and delete the **uv** folder there.
4. Type `%USERPROFILE%\.local\bin` into the address bar and delete **uv.exe**
   and **uvx.exe**.

Your scans and `lungmap_output` folders are not affected. Delete those
yourself if you want to.

---

*A note on accuracy:* lungmap is a research tool, tested on 20 scans. Its
points are measurements to look at and compare, not a medical diagnosis.
