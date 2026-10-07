"""Assignment 03: summarize a telemetry ward's systolic readings.

Run from the assignment directory with the project environment active:

    python3 analysis.py
"""

import numpy as np


def load_readings(filename):
    """Return (patient_ids, monitors, hour_columns, readings) from the supplied CSV.

    readings is a 2D array of integers: one row per patient, one column per
    monitored hour, in the order the header lists them.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    header = lines[0].strip().split(",")
    rows = [line.strip().split(",") for line in lines[1:] if line.strip()]

    patient_ids = np.array([row[0] for row in rows])
    monitors = np.array([row[1] for row in rows])
    hour_columns = np.array(header[2:])
    readings = np.array([row[2:] for row in rows]).astype(int)
    return patient_ids, monitors, hour_columns, readings


def main():
    patient_ids, monitors, hour_columns, readings = load_readings("data/bp_readings.csv")
    print(f"Loaded {readings.shape[0]} patients x {readings.shape[1]} hours")

    print(f"Patient IDs shape: {patient_ids.shape}, first 3: {patient_ids[:3]}")
    print(f"Monitors shape: {monitors.shape}, first 3: {monitors[:3]}")
    print(f"Hour columns shape: {hour_columns.shape}, first 3: {hour_columns[:3]}")
    print(f"Readings shape: {readings.shape}, first 3 rows: {readings[:3]}")

    patients = readings.shape[0]
    total_readings = readings.size
    mean_sbp = readings.mean()
    sd_sbp = readings.std()
    min_sbp = readings.min()
    max_sbp = readings.max()

    print(f"Patients: {patients}, Total readings: {total_readings}, Mean SBP: {mean_sbp}, SD SBP: {sd_sbp}, Min SBP: {min_sbp}, Max SBP: {max_sbp}")

    patient_means = readings.mean(axis=1)
    print(f"Patient means shape: {patient_means.shape}, first 3: {patient_means[:3]}")

    stage2_patients = np.sum(patient_means >= 140)
    print(f"Stage 2 patients: {stage2_patients}")
    highest_patient = patient_ids[patient_means.argmax()]
    print(f"Highest patient: {highest_patient}")
    highest_patient_mean = patient_means.max()
    print(f"Highest patient mean: {highest_patient_mean}")

    hour_means = readings.mean(axis=0)
    print(f"Hour means: {hour_means}")
    peak_hour_column = hour_columns[hour_means.argmax()]
    print(f"Peak hour column: {peak_hour_column}")
    peak_hour_mean = hour_means.max()
    print(f"Peak hour: {peak_hour_column} with mean {peak_hour_mean}")
    
    m01_means = patient_means[monitors == "M01"]
    print(f"M01 means shape: {m01_means.shape}, mean: {m01_means.mean()}")

    monitor_names = sorted(set(monitors))

    monitor_averages = []
    for name in monitor_names:
        monitor_means = patient_means[monitors == name]
        monitor_averages.append(monitor_means.mean())
        print(f"Monitor {name} average: {monitor_means.mean()}")
    
    monitor_averages = np.array(monitor_averages)

    high_monitor = monitor_names[monitor_averages.argmax()]
    print(f"High monitor average: {monitor_averages.max()} ({high_monitor})")

    high_mask = monitors == high_monitor

    high_avg = patient_means[high_mask].mean()
    other_avg = patient_means[~high_mask].mean()
    monitor_offset = high_avg - other_avg
    stage2_other_monitors = np.sum(patient_means[~high_mask] >= 140)
    print(f"High monitor average: {high_avg}, Other monitor average: {other_avg}, Monitor offset: {monitor_offset}, Stage 2 other monitors: {stage2_other_monitors}")

    with open("output/vitals_summary.txt", "w", encoding="utf-8") as file:
        file.write(f"patients: {patients}\n")
        file.write(f"readings: {total_readings}\n")
        file.write(f"mean_sbp: {mean_sbp}\n")
        file.write(f"sd_sbp: {sd_sbp}\n")
        file.write(f"min_sbp: {min_sbp}\n")
        file.write(f"max_sbp: {max_sbp}\n")
        file.write(f"stage2_patients: {stage2_patients}\n")
        file.write(f"highest_patient: {highest_patient}\n")
        file.write(f"highest_patient_mean: {highest_patient_mean}\n")
        file.write(f"peak_hour_column: {peak_hour_column}\n")
        file.write(f"peak_hour_mean: {peak_hour_mean}\n")
        file.write(f"high_monitor: {high_monitor}\n")
        file.write(f"monitor_offset: {monitor_offset}\n")
        file.write(f"stage2_other_monitors: {stage2_other_monitors}\n")


if __name__ == "__main__":
    main()
