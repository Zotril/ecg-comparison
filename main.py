#!/usr/bin/python3
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import scipy.signal as signal
import pathlib
from ecgdetectors import Detectors
#import sys
import os
import pandas as pd
import wfdb
from wfdb.processing import find_peaks, gqrs_detect
#from templates import qrs_templates
import pywt

current_dir = pathlib.Path(__file__).resolve()

example_dir = current_dir.parent/'datasets'/'ecg.tsv'
unfiltered_ecg_dat = np.loadtxt(example_dir) 
unfiltered_ecg = unfiltered_ecg_dat[:, 0]
fs1 = 250
detectors1 = Detectors(fs1)

def filter_raw_ecg1(raw_ecg):
    f1 = 5/fs1
    f2 = 15/fs1
    b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')
    filtered_ecg = signal.lfilter(b, a, raw_ecg)        
    diff = np.diff(filtered_ecg) 
    squared = diff*diff
    return squared

filtered_ecg1 = filter_raw_ecg1(unfiltered_ecg)

#r_peaks1 = detectors1.pan_tompkins_detector(unfiltered_ecg)
#r_peaks1 = detectors1.hamilton_detector(unfiltered_ecg)
r_peaks1 = detectors1.engzee_detector(unfiltered_ecg)
#r_peaks1 = detectors1.two_average_detector(unfiltered_ecg)
#r_peaks1 = detectors1.swt_detector(unfiltered_ecg)
#r_peaks1 = detectors1.wqrs_detector(unfiltered_ecg)
#r_peaks1 = detectors1.christov_detector(unfiltered_ecg)

r_ts1 = np.array(r_peaks1) / fs1

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
t1 = np.linspace(0, len(unfiltered_ecg) / fs1, len(unfiltered_ecg))
plt.plot(t1, unfiltered_ecg)
plt.title("Raw ECG")
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")

plt.subplot(3, 1, 2)
t1 = np.linspace(0, len(filtered_ecg1) / fs1, len(filtered_ecg1))
plt.plot(t1, filtered_ecg1)
plt.title("Filtered ECG")
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")

plt.subplot(3, 1, 3)
t1 = np.linspace(0, len(unfiltered_ecg) / fs1, len(unfiltered_ecg))
plt.plot(t1, unfiltered_ecg)
plt.plot(r_ts1, unfiltered_ecg[r_peaks1], 'ro')
plt.title("Detected R peaks")
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.tight_layout()

intervals = np.diff(r_ts1)
heart_rate = 60.0/intervals
plt.figure()
plt.plot(r_ts1[1:],heart_rate)
plt.title("Heart rate")
plt.xlabel("time/sec")
plt.ylabel("HR/BPM")
plt.ylim([0,200 ])
plt.show()

data_path = r'C:/Users/yorestarii/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
mat_dir = os.path.join(data_path, 'records500')
csv_path = os.path.join(data_path, 'ptbxl_database.csv')
scp_path = os.path.join(data_path, 'scp_statements.csv')

metadata = pd.read_csv(csv_path, index_col='ecg_id')

diagnostic_labels = pd.read_csv(scp_path, index_col=0)

# Function to extract ecg_ids for a specific diagnostic class
def get_ecg_ids_by_class(diagnostic_class):
    codes = diagnostic_labels.index[diagnostic_labels['diagnostic_class'] == diagnostic_class]
    return metadata[metadata['scp_codes'].str.contains('|'.join(codes))].index

# Retrieve IDs for each condition
normal_ids = get_ecg_ids_by_class('NORM')
mi_ids = get_ecg_ids_by_class('MI')
hyp_ids = get_ecg_ids_by_class('HYP')

# Function to load raw ECG signal from a .dat file
def load_ecg(ecg_id):
    # Relative path from metadata
    relative_path = metadata.loc[ecg_id, 'filename_hr']  # e.g., "records500/00000/00001_hr"
    file_path = os.path.join(data_path, relative_path)  # Construct the full path
    print(f"Loading ECG data from: {file_path}")  # Debugging info

    # Load ECG signal using wfdb
    record = wfdb.rdsamp(file_path)  # Returns signal and metadata
    ecg_signal = record[0]  # Raw ECG signal (multi-lead)
    return ecg_signal

# Load one ECG signal for each condition
raw_normal_ecg = load_ecg(normal_ids[0])[:, 0]
raw_mi_ecg = load_ecg(mi_ids[0])[:, 0]
raw_hyp_ecg = load_ecg(hyp_ids[0])[:, 0]

fs = 500
detectors = Detectors(fs)

#-----------------------Pan-Tompkins----------------------------------------

# def filter_raw_ecg(raw_ecg):
#     f1 = 5/fs
#     f2 = 15/fs
#     b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')
#     filtered_ecg = signal.lfilter(b, a, raw_ecg)        
#     diff = np.diff(filtered_ecg) 
#     squared = diff*diff
#     return squared

# filtered_normal_ecg = filter_raw_ecg(raw_normal_ecg)
# filtered_mi_ecg = filter_raw_ecg(raw_mi_ecg)
# filtered_hyp_ecg = filter_raw_ecg(raw_hyp_ecg)

# normal_r_peaks = detectors.pan_tompkins_detector(raw_normal_ecg)
# mi_r_peaks = detectors.pan_tompkins_detector(raw_mi_ecg)
# hyp_r_peaks = detectors.pan_tompkins_detector(raw_hyp_ecg)

#------------------------Hamilton-Tompkins----------------------------------

# def filter_raw_ecg(raw_ecg):
#     f1 = 8/fs
#     f2 = 16/fs
#     b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')
#     partially_filtered_ecg = signal.lfilter(b, a, raw_ecg)
#     diff = abs(np.diff(partially_filtered_ecg))
#     b = np.ones(int(0.08*fs))
#     b = b/int(0.08*fs)
#     a = [1]
#     filtered_ecg = signal.lfilter(b, a, diff)
#     return filtered_ecg

# filtered_normal_ecg = filter_raw_ecg(raw_normal_ecg)
# filtered_mi_ecg = filter_raw_ecg(raw_mi_ecg)
# filtered_hyp_ecg = filter_raw_ecg(raw_hyp_ecg)

# normal_r_peaks = detectors.hamilton_detector(raw_normal_ecg)
# mi_r_peaks = detectors.hamilton_detector(raw_mi_ecg)
# hyp_r_peaks = detectors.hamilton_detector(raw_hyp_ecg)

#-------------------------Engzee----------------------------------

def filter_raw_ecg(raw_ecg):
    f1 = 48/fs
    f2 = 52/fs
    b, a = signal.butter(4, [f1*2, f2*2], btype='bandstop')
    filtered_ecg = signal.lfilter(b, a, raw_ecg)
    diff = np.zeros(len(filtered_ecg))
    for i in range(4, len(diff)):
        diff[i] = filtered_ecg[i]-filtered_ecg[i-4]
    ci = [1,4,6,4,1]        
    low_pass = signal.lfilter(ci, 1, diff)
    low_pass[:int(0.2*fs)] = 0
    return low_pass

filtered_normal_ecg = filter_raw_ecg(raw_normal_ecg)
filtered_mi_ecg = filter_raw_ecg(raw_mi_ecg)
filtered_hyp_ecg = filter_raw_ecg(raw_hyp_ecg)

normal_r_peaks = detectors.engzee_detector(raw_normal_ecg)
mi_r_peaks = detectors.engzee_detector(raw_mi_ecg)
hyp_r_peaks = detectors.engzee_detector(raw_hyp_ecg)

#-----------------------Two average------------------------------------

# def MWA_cumulative(input_array, window_size):
    
#     ret = np.cumsum(input_array, dtype=float)
#     ret[window_size:] = ret[window_size:] - ret[:-window_size]
    
#     for i in range(1,window_size):
#         ret[i-1] = ret[i-1] / i
#     ret[window_size - 1:]  = ret[window_size - 1:] / window_size
    
#     return ret

# def filter_raw_ecg(raw_ecg):
#     f1 = 8/fs
#     f2 = 20/fs
#     b, a = signal.butter(2, [f1*2, f2*2], btype='bandpass')
#     filtered_ecg = signal.lfilter(b, a, raw_ecg)
#     window1 = int(0.12*fs)
#     mwa_qrs = MWA_cumulative(abs(filtered_ecg), window1)

#     window2 = int(0.6*fs)
#     mwa_beat = MWA_cumulative(abs(filtered_ecg), window2)

#     blocks = np.zeros(len(raw_ecg))
#     block_height = np.max(filtered_ecg)

#     for i in range(len(mwa_qrs)):
#         if mwa_qrs[i] > mwa_beat[i]:
#             blocks[i] = block_height
#         else:
#             blocks[i] = 0

#     return blocks

# filtered_normal_ecg = filter_raw_ecg(raw_normal_ecg)
# filtered_mi_ecg = filter_raw_ecg(raw_mi_ecg)
# filtered_hyp_ecg = filter_raw_ecg(raw_hyp_ecg)

# normal_r_peaks = detectors.two_average_detector(raw_normal_ecg)
# mi_r_peaks = detectors.two_average_detector(raw_mi_ecg)
# hyp_r_peaks = detectors.two_average_detector(raw_hyp_ecg)

#----------------------SWT---------------------------------------

# def filter_raw_ecg(raw_ecg):
#     swt_level=3
#     padding = -1
#     for i in range(1000):
#         if (len(raw_ecg)+i)%2**swt_level == 0:
#             padding = i
#             break

#     if padding > 0:
#         raw_ecg = np.pad(raw_ecg, (0, padding), 'edge')
#     elif padding == -1:
#         print("Padding greater than 1000 required\n")    

#     swt_ecg = pywt.swt(raw_ecg, 'db3', level=swt_level)
#     swt_ecg = np.array(swt_ecg)
#     swt_ecg = swt_ecg[0, 1, :]

#     squared = swt_ecg*swt_ecg
#     return squared

# filtered_normal_ecg = filter_raw_ecg(raw_normal_ecg)
# filtered_mi_ecg = filter_raw_ecg(raw_mi_ecg)
# filtered_hyp_ecg = filter_raw_ecg(raw_hyp_ecg)

# normal_r_peaks = detectors.swt_detector(raw_normal_ecg)
# mi_r_peaks = detectors.swt_detector(raw_mi_ecg)
# hyp_r_peaks = detectors.swt_detector(raw_hyp_ecg)

#--------------------WQRS---------------------------------

# def filter_raw_ecg(raw_ecg):
#     nyq = 0.5 * fs
#     order = 2

#     normal_cutoff = 15 / nyq

#     b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
#     y = signal.lfilter(b, a, raw_ecg)
#     return y

# filtered_normal_ecg = filter_raw_ecg(raw_normal_ecg)
# filtered_mi_ecg = filter_raw_ecg(raw_mi_ecg)
# filtered_hyp_ecg = filter_raw_ecg(raw_hyp_ecg)

# normal_r_peaks = detectors.wqrs_detector(raw_normal_ecg)
# mi_r_peaks = detectors.wqrs_detector(raw_mi_ecg)
# hyp_r_peaks = detectors.wqrs_detector(raw_hyp_ecg)
#-----------------Christov--------------------------------------

# def filter_raw_ecg(raw_ecg):
#     total_taps = 0

#     b = np.ones(int(0.02*fs))
#     b = b/int(0.02*fs)
#     total_taps += len(b)
#     a = [1]

#     MA1 = signal.lfilter(b, a, raw_ecg)

#     b = np.ones(int(0.028*fs))
#     b = b/int(0.028*fs)
#     total_taps += len(b)
#     a = [1]

#     MA2 = signal.lfilter(b, a, MA1)

#     Y = []
#     for i in range(1, len(MA2)-1):
        
#         diff = abs(MA2[i+1]-MA2[i-1])

#         Y.append(diff)

#     b = np.ones(int(0.040*fs))
#     b = b/int(0.040*fs)
#     total_taps += len(b)
#     a = [1]

#     MA3 = signal.lfilter(b, a, Y)

#     MA3[0:total_taps] = 0
#     return MA3

# filtered_normal_ecg = filter_raw_ecg(raw_normal_ecg)
# filtered_mi_ecg = filter_raw_ecg(raw_mi_ecg)
# filtered_hyp_ecg = filter_raw_ecg(raw_hyp_ecg)

# normal_r_peaks = detectors.christov_detector(raw_normal_ecg)
# mi_r_peaks = detectors.christov_detector(raw_mi_ecg)
# hyp_r_peaks = detectors.christov_detector(raw_hyp_ecg)

# #---------------------------------------------------------------------

normal_r_ts = np.array(normal_r_peaks) / fs
mi_r_ts = np.array(mi_r_peaks) / fs
hyp_r_ts = np.array(hyp_r_peaks) / fs

# Normal ECG
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
t = np.linspace(0, len(raw_normal_ecg) / fs, len(raw_normal_ecg))
plt.plot(t, raw_normal_ecg)
plt.title('Raw Normal ECG')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()

plt.subplot(3, 1, 2)
t = np.linspace(0, len(filtered_normal_ecg) / fs, len(filtered_normal_ecg))
plt.plot(t, filtered_normal_ecg)
plt.title('Filtered Normal ECG')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()

plt.subplot(3, 1, 3)
t = np.linspace(0, len(raw_normal_ecg) / fs, len(raw_normal_ecg))
plt.plot(t, raw_normal_ecg)
plt.plot(normal_r_ts, raw_normal_ecg[normal_r_peaks], 'ro')
plt.title('Normal detected R peaks')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()
plt.tight_layout()

# Myocardial Infarction ECG
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
t = np.linspace(0, len(raw_mi_ecg) / fs, len(raw_mi_ecg))
plt.plot(t, raw_mi_ecg)
plt.title('Raw Myocardial Infarction ECG')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()

plt.subplot(3, 1, 2)
t = np.linspace(0, len(filtered_mi_ecg) / fs, len(filtered_mi_ecg))
plt.plot(t, filtered_mi_ecg)
plt.title('Myocardial Infarction ECG')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()

plt.subplot(3, 1, 3)
t = np.linspace(0, len(raw_mi_ecg) / fs, len(raw_mi_ecg))
plt.plot(t, raw_mi_ecg)
plt.plot(mi_r_ts, raw_mi_ecg[mi_r_peaks], 'ro')
plt.title('Myocardial Infarction detected R peaks')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()
plt.tight_layout()

# Hypertrophy ECG
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
t = np.linspace(0, len(raw_hyp_ecg) / fs, len(raw_hyp_ecg))
plt.plot(t, raw_hyp_ecg)
plt.title('Raw Hypertrophy ECG')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()

plt.subplot(3, 1, 2)
t = np.linspace(0, len(filtered_hyp_ecg) / fs, len(filtered_hyp_ecg))
plt.plot(t, filtered_hyp_ecg)
plt.title('Raw Hypertrophy ECG')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()

plt.subplot(3, 1, 3)
t = np.linspace(0, len(raw_hyp_ecg) / fs, len(raw_hyp_ecg))
plt.plot(t, raw_hyp_ecg)
plt.plot(hyp_r_ts, raw_hyp_ecg[hyp_r_peaks], 'ro')
plt.title('Hypertrophy detected R peaks')
plt.ylabel("ECG/mV")
plt.xlabel("time/sec")
plt.grid()
plt.tight_layout()

plt.show()
