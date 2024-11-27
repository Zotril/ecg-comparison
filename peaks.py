#!/usr/bin/python3
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

fs = 500
time = np.linspace(0, 10, fs * 10)
signal = []
manual_peaks = []

def onclick(event):
    if event.button == 1:
        x_coord = event.xdata
        y_coord = event.ydata

        closest_time = min(time, key=lambda t: abs(t - x_coord))
        closest_index = np.argmin(np.abs(time - closest_time))
        closest_signal_value = signal[closest_index]

        plt.plot(time[closest_index], closest_signal_value, 'ro')
        manual_peaks.append(closest_time)

        plt.draw()

def annoted_peaks(fs, signal):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, signal, label="ECG Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Click to mark peaks")
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.legend()
    plt.show()

    print("Manual Peaks (in seconds):", manual_peaks)
    print("Manual Peaks (in samples):", [int(p * fs) for p in manual_peaks])
    return manual_peaks