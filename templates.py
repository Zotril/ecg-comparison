#!/usr/bin/python3
# templates.py
qrs_templates = {
    "Normal": [-0.113, -0.105, -0.1, -0.104, -0.119, -0.149, -0.18, -0.174, -0.163, -0.156, -0.148, -0.145, -0.159, -0.185, -0.214, -0.215, -0.175, -0.145, -0.096, -0.068, -0.043, 0.015, 0.041, 0.168, 0.278, 0.307, 0.365, 0.466, 0.543, 0.546, 0.524, 0.49, 0.32, 0.21, 0.142, 0.054, -0.005, -0.012, -0.006, -0.037, -0.066, -0.05, -0.07, -0.108, -0.108, -0.114, -0.165, -0.156, -0.139, -0.131, -0.123, -0.11, -0.099, -0.116, -0.127, -0.154, -0.21, -0.175, -0.126, -0.114, -0.121, -0.12, -0.118, -0.135, -0.151, -0.175, -0.215, -0.205, -0.192, -0.187, -0.177, -0.165, -0.157, -0.148, -0.146, -0.151, -0.15, -0.139, -0.126, -0.13, -0.115, -0.09, -0.182, -0.196, -0.149, -0.144, -0.17, -0.156, -0.138, -0.119, -0.086, -0.065, -0.101, -0.152, -0.164, -0.146, -0.14, -0.133, -0.12, -0.112, -0.1, -0.1, -0.165, -0.164, -0.12, -0.123, -0.165, -0.163, -0.156, -0.151, -0.144, -0.14, -0.137, -0.159, -0.17, -0.157, -0.15, -0.142, -0.134, -0.126, -0.118, -0.11, -0.11, -0.109, -0.115, -0.118, -0.1, -0.084, -0.064, -0.049, -0.026, -0.015, -0.102, -0.129, -0.116, -0.121, -0.125, -0.125, -0.123, -0.129, -0.116, -0.08, -0.105, -0.137, -0.13, -0.097, -0.07, -0.048, -0.043, -0.03]
,
    "MI": [-0.035, -0.04, -0.061, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.07, -0.074, -0.075, -0.075, -0.075, -0.075, -0.075, -0.075, -0.075, -0.075, -0.075, -0.075, -0.075, -0.073, -0.069, -0.065, -0.061, -0.06, -0.058, -0.055, -0.055, -0.043, -0.033, -0.028, -0.023, -0.015, -0.007, -0.002, -0.013, -0.027, -0.025, -0.025, -0.025, -0.027, -0.022, 0.0, 0.021, 0.033, 0.035, 0.035, 0.04, 0.045, 0.009, 0.003, 0.045, 0.07, 0.075, 0.075, 0.058, 0.038, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.042, 0.026, 0.007, -0.002, -0.005, -0.003, 0.01, -0.002, -0.029, -0.035, -0.051, -0.06, -0.06, -0.06, -0.06, -0.06, -0.06, -0.06, -0.06, -0.06, -0.064, -0.065, -0.065, -0.065, -0.065, -0.065, -0.065, -0.064, -0.067, -0.075, -0.076, -0.074, -0.077, -0.08, -0.08, -0.079, -0.083, -0.085, -0.086, -0.09, -0.094, -0.095, -0.094, -0.096, -0.1, -0.1, -0.099, -0.102, -0.105, -0.105, -0.105, -0.108, -0.112, -0.116, -0.12, -0.127, -0.107, -0.089, -0.064, 0.005, 0.091, 0.196, 0.348, 0.521, 0.68, 0.858, 1.004, 1.09, 1.107, 1.075, 1.045, 0.885, 0.707]
,
    "HYP": [0.794, 0.747, 0.613, 0.475, 0.366, 0.213, 0.013, -0.133, -0.175, -0.226, -0.27, -0.282, -0.299, -0.34, -0.326, -0.311, -0.294, -0.27, -0.25, -0.216, -0.206, -0.2, -0.189, -0.185, -0.166, -0.143, -0.13, -0.113, -0.09, -0.099, -0.087, -0.07, -0.079, -0.095, -0.088, -0.083, -0.087, -0.082, -0.065, -0.056, -0.055, -0.053, -0.05, -0.05, -0.042, -0.04, -0.039, -0.034, -0.03, -0.026, -0.022, -0.02, -0.02, -0.02, -0.02, -0.017, -0.014, -0.015, -0.015, -0.007, 0.0, -0.002, -0.005, 0.0, 0.009, 0.011, 0.009, 0.011, 0.015, 0.015, 0.015, 0.015, 0.016, 0.02, 0.02, 0.02, 0.02, 0.021, 0.025, 0.025, 0.024, 0.029, 0.036, 0.04, 0.044, 0.044, 0.048, 0.056, 0.06, 0.065, 0.064, 0.069, 0.075, 0.075, 0.084, 0.085, 0.089, 0.095, 0.095, 0.116, 0.125, 0.142, 0.161, 0.155, 0.159, 0.157, 0.157, 0.16, 0.155, 0.158, 0.169, 0.181, 0.193, 0.205, 0.21, 0.21, 0.21, 0.211, 0.215, 0.218, 0.232, 0.246, 0.251, 0.255, 0.259, 0.262, 0.269, 0.276, 0.275, 0.271, 0.267, 0.263, 0.259, 0.255, 0.25, 0.251, 0.245, 0.232, 0.225, 0.227, 0.208, 0.188, 0.185, 0.18, 0.164, 0.153, 0.145, 0.132, 0.12, 0.112]
}

fs = 500

def extract_qrs_template(signal, r_peaks, window_ms=300):
    template_window = int((window_ms / 1000) * fs)  # Convert ms to samples
    qrs_templates = []
    for peak in r_peaks:
        start = max(0, peak - template_window // 2)
        end = min(len(signal), peak + template_window // 2)
        segment = signal[start:end]
        if len(segment) == template_window:
            qrs_templates.append(segment)
    return qrs_templates[0] if qrs_templates else None

# Process signals for each condition
# conditions = {"Normal": raw_normal_ecg, "MI": raw_mi_ecg, "HYP": raw_hyp_ecg}
# templates = {}

# for condition, raw_signal in conditions.items():
#     lead_signal = raw_signal[:, 0]  # Choose Lead I (column 0)
#     r_peaks = detectors.pan_tompkins_detector(lead_signal)  # Detect R-peaks
#     qrs_template = extract_qrs_template(lead_signal, r_peaks)
    
#     if qrs_template is not None:
#         templates[condition] = qrs_template
#         # Plot the QRS template
#         plt.figure(figsize=(10, 5))
#         plt.plot(qrs_template, label=f"{condition} QRS Template (500 Hz)")
#         plt.title(f"{condition} QRS Template")
#         plt.xlabel("Sample Index")
#         plt.ylabel("Amplitude")
#         plt.legend()
#         plt.grid()
#         plt.show()
#         print(f"Extracted {condition} QRS Template:")
#         print(qrs_template)
#     else:
#         print(f"No QRS template extracted for {condition}.")
# Save templates to files
# np.savetxt("normal_template.txt", qrs_templates["Normal"])
# np.savetxt("mi_template.txt", qrs_templates["MI"])
# np.savetxt("hyp_template.txt", qrs_templates["HYP"])