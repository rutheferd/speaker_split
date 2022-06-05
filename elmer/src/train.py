import os
import shutil
from click import password_option
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import logging

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"
SHUFFLE_SEED = 43
SCALE = 0.5


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(
        lambda x: path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE
    )
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(
            tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64
        )
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


def build_dataset(data_audio_path, val_split, batch_size):
    class_names = os.listdir(data_audio_path)
    print(
        "Our class names: {}".format(
            class_names,
        )
    )

    audio_paths = []
    labels = []
    for label, name in enumerate(class_names):
        print(
            "Processing speaker {}".format(
                name,
            )
        )
        dir_path = Path(data_audio_path) / name
        speaker_sample_paths = [
            os.path.join(dir_path, filepath)
            for filepath in os.listdir(dir_path)
            if filepath.endswith(".wav")
        ]
        audio_paths += speaker_sample_paths
        labels += [label] * len(speaker_sample_paths)

    print(
        "Found {} files belonging to {} classes.".format(
            len(audio_paths), len(class_names)
        )
    )

    # Shuffle
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(labels)

    # Split into training and validation
    num_val_samples = int(val_split * len(audio_paths))
    print(
        "Using {} files for training.".format(
            len(audio_paths) - num_val_samples
        )
    )
    train_audio_paths = audio_paths[:-num_val_samples]
    train_labels = labels[:-num_val_samples]

    print("Using {} files for validation.".format(num_val_samples))
    valid_audio_paths = audio_paths[-num_val_samples:]
    valid_labels = labels[-num_val_samples:]

    # Create 2 datasets, one for training and the other for validation
    train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
    train_ds = train_ds.shuffle(
        buffer_size=batch_size * 8, seed=SHUFFLE_SEED
    ).batch(batch_size)

    valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(
        32
    )

    # Add noise to the training set
    # train_ds = train_ds.map(
    #     lambda x, y: (add_noise(x, noises, scale=SCALE), y),
    #     num_parallel_calls=tf.data.AUTOTUNE,
    # )

    # Transform audio wave to the frequency domain using `audio_to_fft`
    train_ds = train_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    valid_ds = valid_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    pass


def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)


def train(train_ds, valid_ds, sampling_rate, num_epochs, class_names):
    model = build_model((sampling_rate // 2, 1), len(class_names))

    model.summary()

    # Compile the model using Adam's default learning rate
    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Add callbacks:
    # 'EarlyStopping' to stop training when the model is not enhancing anymore
    # 'ModelCheckPoint' to always keep the model that has the best val_accuracy
    model_save_filename = "model.h5"

    earlystopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
        model_save_filename, monitor="val_accuracy", save_best_only=True
    )

    history = model.fit(
        train_ds,
        epochs=num_epochs,
        validation_data=valid_ds,
        callbacks=[earlystopping_cb, mdlcheckpoint_cb],
    )

    return history, model


def evaluate_model(model, valid_ds, verbose=True):
    """Return the evaluation data."""
    if verbose:
        logging.info(model.evaluate(valid_ds))
    return model.evaluate(valid_ds)


def run(data_path, val_split, sampling_rate, batch_size, num_epochs):
    sentence = train(
        data_path, val_split, sampling_rate, batch_size, num_epochs
    )
    data_audio_path = os.path.join(data_path, AUDIO_SUBFOLDER)
    data_noise_path = os.path.join(data_path, NOISE_SUBFOLDER)
    print(sentence)
