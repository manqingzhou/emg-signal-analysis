# emg-signal-analysis
Predict loads in potential patient's hand
> *Copyright 2020 Manqing Zhou*

### Table of Contents

- [Preprocessing](#preprocessing)
- [Smoothing](#smoothing)
- [Feature extraction](#feature-extraction)
- [Normalization](#normalization)
- [Multi-Classification model](#multi-classification-model)

## Preprocessing
In order to remove any of low frequency noises from out EMG signal data. Done by a high pass filter to remove any of the non-zero dc levels from the signal data

## Smoothing
Apply a digital smoothing algorithm that outlines the mean trend of the signal develoment

## Feature extraction
Extract the average value of 100 samples to reduce the input data of our classification. Here, I used RMS

## Normalization
Use the sub maximal normalization. The main idea is to divide the data by max rms

## Multi-Classification model
Prediction Part



