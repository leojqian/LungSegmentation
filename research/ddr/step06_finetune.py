# Step 6: fine-tune the Montgomery U-Net (best_model.h5) on the 20-case DDR set.
# Writes best_model_ddr_finetuned_phase3.h5 -- the model lungmap ships -- and
# outputs/ddr/finetune_report.json. Takes several hours on CPU.
#
# Method (rationale: docs/design/2026-08-24-ddr-domain-adaptation.md):
#   1. Preprocessing chosen by a calibration pass on the unmodified base model:
#      min/max windowing, opening kernel 5, the vendor-processed images_pres DICOMs.
#   2. 5-fold cross-validation grouped by patient (16 train / 4 held out).
#   3. Train only on LungArea_truth frames (~2 per case), augmented 5x with small
#      rotations/shifts/contrast changes -- never left-right flips.
#   4. Fine-tune the decoder only; tuning the encoder too wiped out the
#      Montgomery performance within an epoch.
#   5. Keep an epoch only if Montgomery Dice stays >= 0.96; among those, keep
#      the lowest DDR validation loss.
#   6. Save the fold with the best held-out DDR Dice. step07_evaluate.py scores it.
#
# Run from the repo root:
#   uv run --extra research python research/ddr/step06_finetune.py

import glob
import json
import os
import re

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from lungmap.formats.dicom_io import (load_frames, parse_lung_area_truth, to_model_input,
                                     window_params)
from lungmap.segmentation.pipeline import segment   # always called with MASK_OPEN_KERNEL

DDR_DIR = "SampleDDR_August2026"
IMAGES_PRES_DIR = os.path.join(DDR_DIR, "images_pres")
LUNGAREA_DIR = os.path.join(DDR_DIR, "LungArea_truth")
DMMODE_DIR = os.path.join(DDR_DIR, "DM-MODE_truth")

BASE_MODEL = "best_model.h5"
MODEL_OUT = "best_model_ddr_finetuned_phase3.h5"
OUT_DIR = "outputs/ddr"

# --- winning Phase 1 preprocessing (see module docstring) --------------------
USE_DICOM_WINDOW_TAGS = False   # False = min/max normalization
MASK_OPEN_KERNEL = 5            # morphological-opening kernel, post-threshold
MODEL_SIZE = 256                # the U-Net's native input resolution

# --- fine-tuning recipe --------------------------------------------------------
MONTGOMERY_DICE_FLOOR = 0.96
N_FOLDS = 5
SEED = 42
LEARNING_RATE = 1e-5
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 8
BATCH_SIZE = 4
AUGMENTATIONS_PER_IMAGE = 4     # + the original = 5x the training set
DECODER_START_LAYER = "up_sampling2d"  # first decoder layer in best_model.h5


# --- case discovery ------------------------------------------------------------

def find_cases():
    """SampleDDR_August2026/images_pres -> [(case_id, dcm_path, xml_path, raw_path), ...]

    Only cases with all three files (image + both ground-truth formats) present.
    """
    cases = []
    pattern = re.compile(r"(.+)LogPlusRegius\.dcm$")
    for dcm_path in sorted(glob.glob(os.path.join(IMAGES_PRES_DIR, "*.dcm"))):
        m = pattern.match(os.path.basename(dcm_path))
        if not m:
            continue
        case_id = m.group(1)
        xml_path = os.path.join(DMMODE_DIR, f"{case_id}_truth.xml")
        raw_path = os.path.join(LUNGAREA_DIR, f"{case_id}_truth.raw")
        if os.path.exists(xml_path) and os.path.exists(raw_path):
            cases.append((case_id, dcm_path, xml_path, raw_path))
    return cases


def dice(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    total = pred.sum() + truth.sum()
    return 2 * np.logical_and(pred, truth).sum() / total if total else 1.0


# --- Montgomery regression guard ----------------------------------------------

def load_montgomery_validation():
    imgs = np.load("data/Unet-Validate-Lung-Images.npy")
    masks = np.load("data/Unet-Validate-Lung-Masks.npy")
    return imgs.astype(np.float32), masks.astype(bool)


class FloorGatedCheckpoint(tf.keras.callbacks.Callback):
    """Every epoch: score Montgomery Dice; remember the lowest-fold-val-loss
    epoch among those meeting MONTGOMERY_DICE_FLOOR. See module docstring."""

    def __init__(self, montgomery_imgs, montgomery_masks, floor=MONTGOMERY_DICE_FLOOR):
        super().__init__()
        self.montgomery_imgs = montgomery_imgs
        self.montgomery_masks = montgomery_masks
        self.floor = floor
        self.trajectory = []
        self.best = None  # (val_loss, weights, epoch, montgomery_dice)

    def _montgomery_dice(self):
        pred = self.model.predict(self.montgomery_imgs, verbose=0)[..., 0] > 0.5
        inter = np.logical_and(pred, self.montgomery_masks).sum(axis=(1, 2))
        total = pred.sum(axis=(1, 2)) + self.montgomery_masks.sum(axis=(1, 2))
        return float((2 * inter / total).mean())

    def on_epoch_end(self, epoch, logs=None):
        dice_score = self._montgomery_dice()
        val_loss = float(logs.get("val_loss"))
        self.trajectory.append({"epoch": epoch, "montgomery_dice": dice_score, "val_loss": val_loss})
        if dice_score >= self.floor and (self.best is None or val_loss < self.best[0]):
            self.best = (val_loss, [w.copy() for w in self.model.get_weights()], epoch, dice_score)


# --- augmentation --------------------------------------------------------------

def augment(image, mask, rng):
    """Small rotation + translation + brightness/contrast jitter. No flip —
    chest anatomy is not left-right symmetric."""
    h, w = image.shape
    angle = rng.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += rng.uniform(-0.05, 0.05) * w
    M[1, 2] += rng.uniform(-0.05, 0.05) * h

    image_aug = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask_aug = cv2.warpAffine(mask.astype(np.float32), M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask_aug = (mask_aug > 0.5).astype(np.uint8)

    contrast, brightness = rng.uniform(0.85, 1.15), rng.uniform(-0.05, 0.05)
    image_aug = np.clip(image_aug * contrast + brightness, 0, 1).astype(np.float32)
    return image_aug, mask_aug


# --- fold data -----------------------------------------------------------------

def load_fold_frames(cases):
    """cases -> (images single-channel (N,256,256) float32, masks (N,256,256) uint8).

    One entry per LungArea_truth-annotated frame across all given cases.
    """
    images, masks = [], []
    for case_id, dcm_path, xml_path, raw_path in cases:
        frames, spacing_mm, photometric = load_frames(dcm_path)
        truth = parse_lung_area_truth(raw_path, shape=frames.shape[1:])
        wc, ww = window_params(dcm_path) if USE_DICOM_WINDOW_TAGS else (None, None)
        for frame_idx, truth_mask in truth.items():
            norm = to_model_input(frames[frame_idx], photometric, wc, ww)
            images.append(cv2.resize(norm, (MODEL_SIZE, MODEL_SIZE)))
            masks.append(cv2.resize(truth_mask.astype(np.uint8), (MODEL_SIZE, MODEL_SIZE),
                                    interpolation=cv2.INTER_NEAREST))
    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.uint8)


def build_training_set(images, masks, seed):
    """Originals plus AUGMENTATIONS_PER_IMAGE augmented copies each -> (X, y) 3-channel."""
    rng = np.random.default_rng(seed)
    all_images, all_masks = list(images), list(masks)
    for image, mask in zip(images, masks):
        for _ in range(AUGMENTATIONS_PER_IMAGE):
            aug_img, aug_mask = augment(image, mask, rng)
            all_images.append(aug_img)
            all_masks.append(aug_mask)
    X = np.repeat(np.array(all_images)[..., None], 3, axis=-1)
    y = np.array(all_masks, dtype=np.float32)[..., None]
    return X, y


# --- one fold --------------------------------------------------------------

def freeze_encoder(model, start_layer=DECODER_START_LAYER):
    """Freeze every layer up to (not including) start_layer -- see DECODER_START_LAYER."""
    names = [l.name for l in model.layers]
    cutoff = names.index(start_layer)
    for layer in model.layers[:cutoff]:
        layer.trainable = False
    return model


def fine_tune_fold(fold_idx, train_cases, val_cases, montgomery):
    train_images, train_masks = load_fold_frames(train_cases)
    val_images, val_masks = load_fold_frames(val_cases)
    X_train, y_train = build_training_set(train_images, train_masks, SEED + fold_idx)
    X_val = np.repeat(val_images[..., None], 3, axis=-1)
    y_val = val_masks[..., None].astype(np.float32)

    model = freeze_encoder(tf.keras.models.load_model(BASE_MODEL))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])

    checkpoint = FloorGatedCheckpoint(*montgomery)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, verbose=0,
             validation_data=(X_val, y_val), shuffle=True,
             callbacks=[checkpoint,
                       tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                        patience=EARLY_STOP_PATIENCE)])

    if checkpoint.best is None:
        print(f"  fold {fold_idx}: no epoch met the {MONTGOMERY_DICE_FLOOR} Montgomery floor "
              f"— falling back to unmodified {BASE_MODEL}")
        return tf.keras.models.load_model(BASE_MODEL), None

    val_loss, weights, epoch, montgomery_dice = checkpoint.best
    model.set_weights(weights)
    print(f"  fold {fold_idx}: kept epoch {epoch} (val_loss={val_loss:.4f}, "
         f"montgomery_dice={montgomery_dice:.4f})")

    # this fold's own held-out DDR Dice, for checkpoint selection across folds
    fold_dice = []
    for case_id, dcm_path, xml_path, raw_path in val_cases:
        frames, _, photometric = load_frames(dcm_path)
        truth = parse_lung_area_truth(raw_path, shape=frames.shape[1:])
        wc, ww = window_params(dcm_path) if USE_DICOM_WINDOW_TAGS else (None, None)
        for frame_idx, truth_mask in truth.items():
            img = to_model_input(frames[frame_idx], photometric, wc, ww)
            fold_dice.append(dice(segment(img, model, open_kernel=MASK_OPEN_KERNEL), truth_mask))

    return model, {"epoch": epoch, "montgomery_dice": montgomery_dice,
                   "fold_dice": float(np.mean(fold_dice))}


# --- driver --------------------------------------------------------------------

def main():
    cases = find_cases()
    if not cases:
        raise SystemExit(f"no cases found under {IMAGES_PRES_DIR}")
    case_ids = [c[0] for c in cases]
    print(f"{len(cases)} DDR cases, recipe: lr={LEARNING_RATE} encoder frozen -> {MODEL_OUT}")

    montgomery = load_montgomery_validation()
    kf = KFold(n_splits=min(N_FOLDS, len(cases)), shuffle=True, random_state=SEED)

    fold_assignment = {}
    fold_reports = []
    best_fold_dice = None

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(case_ids)):
        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]
        for c in val_cases:
            fold_assignment[c[0]] = fold_idx
        print(f"\nfold {fold_idx}: train={[c[0] for c in train_cases]} "
             f"val={[c[0] for c in val_cases]}")

        model, checkpoint_info = fine_tune_fold(fold_idx, train_cases, val_cases, montgomery)
        fold_reports.append({"fold": fold_idx, "checkpoint": checkpoint_info,
                            "val_cases": [c[0] for c in val_cases]})

        # the deployed model: whichever floor-passing fold has the best held-out
        # DDR Dice — not the fold with the highest Montgomery Dice, which is
        # only the regression guard.
        fold_dice = checkpoint_info["fold_dice"] if checkpoint_info else None
        if fold_dice is not None and (best_fold_dice is None or fold_dice > best_fold_dice):
            best_fold_dice = fold_dice
            model.save(MODEL_OUT)
            print(f"  saved as {MODEL_OUT} (fold {fold_idx} DDR Dice={fold_dice:.3f}, best so far)")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "fold_assignment.json"), "w") as f:
        json.dump(fold_assignment, f, indent=2)
    with open(os.path.join(OUT_DIR, "finetune_report.json"), "w") as f:
        json.dump(fold_reports, f, indent=2)

    fold_dices = [r["checkpoint"]["fold_dice"] for r in fold_reports if r["checkpoint"]]
    n_fallback = sum(1 for r in fold_reports if r["checkpoint"] is None)
    print(f"\nmean out-of-fold DDR Dice: {np.mean(fold_dices):.3f}  ({len(fold_dices)}/{len(fold_reports)} folds)")
    print(f"{n_fallback}/{len(fold_reports)} folds fell back to unmodified {BASE_MODEL} "
         f"(no epoch met the {MONTGOMERY_DICE_FLOOR} Montgomery floor)")
    print(f"wrote {OUT_DIR}/fold_assignment.json, finetune_report.json")
    print(f"deployed model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
