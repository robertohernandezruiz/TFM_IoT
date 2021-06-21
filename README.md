# TFM_IoT
TFM IoT - ECG LSTM Authentication

- ECG_biometrics.m: main code to test proposed algorithm with a ECG-signals dataset.
- ReadECGData.m: code that reads ECG signals from .mat files in "Sujetos" directory and creates ECGSignalsData.mat variable with Signals and associated Classes.
- segmentSignals.m: Matlab native script to segment signals in smaller windows with standarized 9000 samples each one.
- dataset.csv: CSV file that contains the names of the .mat files in "Sujetos" to be processed and their associated label/class.
