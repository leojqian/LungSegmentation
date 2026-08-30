# Phase 2 fine-tuning for the DDR domain-adaptation spec
# (docs/superpowers/specs/2026-08-24-ddr-domain-adaptation-design.md).
#
# 5-fold group cross-validation by case (16 train / 4 val patients per fold,
# seed 42). Out-of-fold predictions across all 5 folds give one Dice/point-error
# number directly comparable to Phase 1's — every case is scored by a model that
# never saw it during training.
#
# Checkpoint selection is decoupled from early stopping: a 4-case fold's own
# validation loss is too noisy to trust for picking which epoch's weights to
# keep, so early stopping is compute-saving only, and the checkpoint kept is
# whichever epoch has the lowest fold validation loss AMONG epochs where
# Montgomery Dice stays >= 0.96 (the pre-registered regression floor). A fold
# where no epoch ever meets the floor falls back to unmodified best_model.h5 —
# reported explicitly, not hidden.
#
# Reads Phase 1's winning preprocessing (windowing/kernel/image source) from
# outputs/ddr_calibration/findings.json — run Step7CalibrateDDR.py first.

import argparse
import csv
import json
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

import Step6ValidateDDR as step6
import Step7CalibrateDDR as step7

OUT_DIR = "outputs/ddr"
CALIBRATION_FILE = "outputs/ddr_calibration/findings.json"
MONTGOMERY_DICE_FLOOR = 0.96
N_FOLDS = 5
SEED = 42
LEARNING_RATE = 1e-5
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 8
BATCH_SIZE = 4
AUGMENTATIONS_PER_IMAGE = 4  # + the original = 5x the training set

# Phase 3 (--freeze-encoder / --lr): the design doc's Phase 2 addendum found
# all-layer fine-tuning at LEARNING_RATE forgets Montgomery within epoch 0-1 in
# every fold, so no checkpoint ever meets MONTGOMERY_DICE_FLOOR. "up_sampling2d"
# is the first decoder layer (the bottleneck's upsample) in best_model.h5's
# architecture -- freezing everything before it restricts adaptation to the
# decoder, which should forget more slowly.
DECODER_START_LAYER = "up_sampling2d"

FIELDNAMES = step6.FIELDNAMES + ["fold", "checkpoint_epoch", "checkpoint_montgomery_dice"]


# --- preprocessing config, from Phase 1 --------------------------------------

def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        raise SystemExit(f"{CALIBRATION_FILE} not found — run Step7CalibrateDDR.py first")
    with open(CALIBRATION_FILE) as f:
        winner = json.load(f)["winner"]
    return winner["window"], winner["kernel"], winner["source_raw"]


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
        dice = self._montgomery_dice()
        val_loss = float(logs.get("val_loss"))
        self.trajectory.append({"epoch": epoch, "montgomery_dice": dice, "val_loss": val_loss})
        if dice >= self.floor and (self.best is None or val_loss < self.best[0]):
            self.best = (val_loss, [w.copy() for w in self.model.get_weights()], epoch, dice)


# --- augmentation --------------------------------------------------------------

def augment(image, mask, rng):
    """Small rotation + translation + brightness/contrast jitter. No flip —
    chest anatomy is not left-right symmetric (see design doc)."""
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

def load_fold_frames(cases, window):
    """cases -> (images single-channel (N,256,256) float32, masks (N,256,256) uint8).

    One entry per LungArea_truth-annotated frame across all given cases.
    """
    images, masks = [], []
    for case_id, dcm_path, xml_path, raw_path in cases:
        frames, spacing_mm, photometric = step6.load_frames(dcm_path)
        truth = step6.parse_lung_area_truth(raw_path, shape=frames.shape[1:])
        wc, ww = step6.window_params(dcm_path) if window else (None, None)
        for frame_idx, truth_mask in truth.items():
            norm = step6.to_model_input(frames[frame_idx], photometric, wc, ww)
            images.append(cv2.resize(norm, (step6.MODEL_SIZE, step6.MODEL_SIZE)))
            masks.append(cv2.resize(truth_mask.astype(np.uint8),
                                    (step6.MODEL_SIZE, step6.MODEL_SIZE),
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


def fine_tune_fold(fold_idx, train_cases, val_cases, window, montgomery, seed=SEED,
                   lr=LEARNING_RATE, freeze_enc=False):
    train_images, train_masks = load_fold_frames(train_cases, window)
    val_images, val_masks = load_fold_frames(val_cases, window)
    X_train, y_train = build_training_set(train_images, train_masks, seed + fold_idx)
    X_val = np.repeat(val_images[..., None], 3, axis=-1)
    y_val = val_masks[..., None].astype(np.float32)

    model = tf.keras.models.load_model(step6.MODEL_FILE)
    if freeze_enc:
        model = freeze_encoder(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])

    checkpoint = FloorGatedCheckpoint(*montgomery)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, verbose=0,
             validation_data=(X_val, y_val), shuffle=True,
             callbacks=[checkpoint,
                       tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                        patience=EARLY_STOP_PATIENCE)])

    if checkpoint.best is None:
        print(f"  fold {fold_idx}: no epoch met the {MONTGOMERY_DICE_FLOOR} Montgomery floor "
              f"— falling back to unmodified {step6.MODEL_FILE}")
        return tf.keras.models.load_model(step6.MODEL_FILE), None, checkpoint.trajectory

    val_loss, weights, epoch, dice = checkpoint.best
    model.set_weights(weights)
    print(f"  fold {fold_idx}: kept epoch {epoch} (val_loss={val_loss:.4f}, "
         f"montgomery_dice={dice:.4f})")
    return model, {"epoch": epoch, "montgomery_dice": dice}, checkpoint.trajectory


# --- driver --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="only use the first N cases (smoke-testing)")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="Phase 3: freeze everything before the decoder (see DECODER_START_LAYER)")
    ap.add_argument("--lr", type=float, default=LEARNING_RATE,
                    help=f"optimizer learning rate (default {LEARNING_RATE})")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help=f"output directory for reports/results (default {OUT_DIR})")
    ap.add_argument("--model-out", default="best_model_ddr_finetuned.h5",
                    help="where to save the best-fold checkpoint")
    args = ap.parse_args()
    out_dir = args.out_dir

    window, kernel, source_raw = load_calibration()
    print(f"Phase 1 winning config: window={window} kernel={kernel} source_raw={source_raw}")
    print(f"recipe: lr={args.lr} freeze_encoder={args.freeze_encoder} -> {args.model_out}")

    cases = step6.find_cases(step7.RAW_IMAGES_DIR if source_raw else step6.IMAGES_PRES_DIR,
                             suffix="" if source_raw else "LogPlusRegius")
    if args.limit:
        cases = cases[:args.limit]
    case_ids = [c[0] for c in cases]

    montgomery = load_montgomery_validation()

    kf = KFold(n_splits=min(N_FOLDS, len(cases)), shuffle=True, random_state=SEED)
    fold_assignment = {}
    all_rows = []
    fold_reports = []
    best_fold_dice = None

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(case_ids)):
        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]
        for c in val_cases:
            fold_assignment[c[0]] = fold_idx
        print(f"\nfold {fold_idx}: train={[c[0] for c in train_cases]} "
             f"val={[c[0] for c in val_cases]}")

        model, checkpoint_info, trajectory = fine_tune_fold(fold_idx, train_cases, val_cases,
                                                             window, montgomery,
                                                             lr=args.lr, freeze_enc=args.freeze_encoder)
        fold_reports.append({"fold": fold_idx, "checkpoint": checkpoint_info,
                            "trajectory": trajectory,
                            "val_cases": [c[0] for c in val_cases]})

        fold_rows = []
        for case_id, dcm_path, xml_path, raw_path in val_cases:
            rows = step6.process_case(case_id, dcm_path, xml_path, raw_path, model,
                                      write_overlays=False, open_kernel=kernel, window=window)
            for row in rows:
                row["fold"] = fold_idx
                row["checkpoint_epoch"] = checkpoint_info["epoch"] if checkpoint_info else None
                row["checkpoint_montgomery_dice"] = (checkpoint_info["montgomery_dice"]
                                                     if checkpoint_info else None)
            fold_rows.extend(rows)
        all_rows.extend(fold_rows)

        fold_dice_values = [r["dice"] for r in fold_rows
                           if r.get("kind") == "lung_area" and "dice" in r]
        fold_dice = float(np.mean(fold_dice_values)) if fold_dice_values else None
        fold_reports[-1]["fold_ddr_dice"] = fold_dice

        # "the" saved fine-tuned model: the fold whose held-out DDR Dice is best
        # among folds that passed the Montgomery floor at all — not the fold
        # with the highest Montgomery Dice, which is only the regression guard.
        if checkpoint_info is not None and fold_dice is not None:
            if best_fold_dice is None or fold_dice > best_fold_dice:
                best_fold_dice = fold_dice
                model.save(args.model_out)
                print(f"  saved as {args.model_out} "
                     f"(fold {fold_idx} DDR Dice={fold_dice:.3f}, best so far)")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fold_assignment.json"), "w") as f:
        json.dump(fold_assignment, f, indent=2)
    with open(os.path.join(out_dir, "finetune_report.json"), "w") as f:
        json.dump(fold_reports, f, indent=2)
    with open(os.path.join(out_dir, "finetune_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    step6._report(all_rows)
    n_fallback = sum(1 for r in fold_reports if r["checkpoint"] is None)
    print(f"\n{n_fallback}/{len(fold_reports)} folds fell back to unmodified {step6.MODEL_FILE} "
         f"(no epoch met the {MONTGOMERY_DICE_FLOOR} Montgomery floor)")
    print(f"wrote {out_dir}/fold_assignment.json, finetune_report.json, finetune_results.csv")


if __name__ == "__main__":
    main()
