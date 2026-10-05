# Step 3: train the U-Net on step 1's arrays.
# Writes best_model.h5, the Montgomery base model that step 6 fine-tunes.
#
# Run from the repo root:
#   uv run --extra research python research/montgomery/step03_train_unet.py

import numpy as np

#load data
print("start loading...")
allImagesNP = np.load("data/Unet-Train-Lung-Images.npy")
maskImagesNP = np.load("data/Unet-Train-Lung-Masks.npy")

allValidateImagesNP = np.load("data/Unet-Validate-Lung-Images.npy")
maskValidateImagesNP = np.load("data/Unet-Validate-Lung-Masks.npy")

print(allImagesNP.shape)
print(maskImagesNP.shape)
print(allValidateImagesNP.shape)
print(maskValidateImagesNP.shape)

Height = 256
Width = 256

import tensorflow as tf
from step02_unet_model import build_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

shape=(256, 256, 3)
lr = 1e-4
batchSize = 4
epochs = 50

model = build_model(shape)
print(model.summary())

opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

stepsPerEpoch = int(np.ceil(len(allImagesNP) / batchSize))
validationSteps = int(np.ceil(len(allValidateImagesNP) / batchSize))

best_model_file = "best_model.h5"
callbacks = [
    ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=20, verbose=1)
]

history = model.fit(allImagesNP, maskImagesNP,
                    batch_size = batchSize,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(allValidateImagesNP, maskValidateImagesNP),
                    validation_steps= validationSteps,
                    steps_per_epoch = stepsPerEpoch,
                    shuffle=True,
                    callbacks=callbacks  )
