"""
This file is adapted from https://github.com/kentaroy47/ESC-50.pytorch/blob/master/train.ipynb
"""
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# DATA PROCESSING
# load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x, fs


# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp


# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()


# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()


# DATA AUGMENTATION
# data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate * np.random.randn(len(x))


# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x) // rate))


# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


# Dataset class
class ESC50Dataset(Dataset):
    def __init__(self, data, label, audio_dir, data_aug=False, _type="train"):
        self.label = label
        self.data_aug = data_aug
        self.data = data
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        x, fs = load_wave_data(self.audio_dir, self.data[idx])

        # augumentations in wave domain.
        if self.data_aug:
            r = np.random.rand()
            if r < 0.3:
                x = add_white_noise(x)

            r = np.random.rand()
            if r < 0.3:
                x = shift_sound(x, rate=1 + np.random.rand())

            r = np.random.rand()
            if r < 0.3:
                x = stretch_sound(x, rate=0.8 + np.random.rand() * 0.4)

        # convert to melsp
        melsp = calculate_melsp(x)

        # normalize
        mean = np.mean(melsp)
        std = np.std(melsp)

        melsp -= mean
        melsp /= std

        melsp = np.asarray([melsp, melsp, melsp])

        return melsp.astype("float32"), label


def get_train_test_dataloader(base_dir, batch_size, num_workers=8, test_fold=1):
    # define directories
    esc_dir = os.path.join(base_dir, "ESC-50-master")
    meta_file = os.path.join(esc_dir, "meta/esc50.csv")
    audio_dir = os.path.join(esc_dir, "audio/")

    # load metadata
    meta_data = pd.read_csv(meta_file)

    # get training dataset and target dataset
    meta_train = meta_data[meta_data["fold"] != test_fold]
    meta_test = meta_data[meta_data["fold"] == test_fold]
    x_train = list(meta_train.loc[:, "filename"])
    y_train = list(meta_train.loc[:, "target"])
    x_test = list(meta_test.loc[:, "filename"])
    y_test = list(meta_test.loc[:, "target"])
    print("Train: {} samples, Test: {} samples".format(len(y_train), len(y_test)))

    # get dataloader
    traindataset = ESC50Dataset(x_train, y_train, audio_dir, data_aug=True)
    testdataset = ESC50Dataset(x_test, y_test, audio_dir, data_aug=False)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    print("Finish creating dataloaders")

    return train_loader, test_loader
