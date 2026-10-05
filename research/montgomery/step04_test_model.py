# Step 4: sanity-check best_model.h5 on one Montgomery image: prints the landmark
# points and shows the mask, diaphragm curves and landmarks in windows.
#
# Run from the repo root:
#   uv run --extra research python research/montgomery/step04_test_model.py

import cv2
import tensorflow as tf

from lungmap.geometry.diaphragm import apex_is_lower_bound
from lungmap.segmentation.pipeline import analyze
from lungmap.rendering.render import display_scale, draw_overlay

MODEL_FILE = "best_model.h5"
TEST_IMAGE = "MontgomerySet/CXR_png/MCUCXR_0035_0.png"

model = tf.keras.models.load_model(MODEL_FILE)
print(model.summary())

# one call: segment, split into lungs, trace each diaphragm, and find the
# corner/dome landmark points — see pipeline.py
result = analyze(TEST_IMAGE, model)
print("predicted mask pixels:", int(result.mask.sum()))

for side in ("R", "L"):
    lung = result.lungs[side]
    if lung is None:
        print(f"  {side}: landmarks not found")
        continue
    for name in ("top", "lower_left", "lower_right"):
        print(f"  {side}-{name:11s} {getattr(lung, name)}")

apex_heights = {}
for side in ("R", "L"):
    curve, lung = result.curves[side], result.lungs[side]
    if curve is None or lung is None:
        print(f"  {side}: dome not found")
        continue
    apex = lung.dome
    floor = apex_is_lower_bound(curve.measured, apex)
    apex_heights[side] = apex
    print(f"  {side}-dome apex   {tuple(apex)}"
          f"{'  (LOWER BOUND - dome hidden, apex at curve end)' if floor else ''}")

if len(apex_heights) == 2:
    # on a PA film the patient's RIGHT hemidiaphragm normally sits higher,
    # pushed up by the liver. R is the image-left lung.
    drop = int(apex_heights["L"][1] - apex_heights["R"][1])
    higher = "R" if drop > 0 else "L"
    print(f"  hemidiaphragm difference: {abs(drop)} px, {higher} sits higher")

cxr = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
f = display_scale(cxr.shape)
small = cv2.resize(cxr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
mask_view = cv2.resize((result.mask * 255).astype("uint8"), None, fx=f, fy=f,
                       interpolation=cv2.INTER_AREA)
overlay = draw_overlay(cxr, result.curves["R"], result.curves["L"],
                       result.masks["R"], result.masks["L"],
                       show_masks=True, show_landmarks=True,
                       left_lung=result.lungs["R"], right_lung=result.lungs["L"])

cv2.imshow("original", small)
cv2.imshow("predicted mask", mask_view)
cv2.imshow("diaphragm + landmarks", overlay)

# NOTE: waitKey only responds to the KEYBOARD - clicking the window's red X does
# nothing and the script stays blocked here. Click a window, then press any key.
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)   # macOS needs one more event-loop tick to actually dismiss the windows
