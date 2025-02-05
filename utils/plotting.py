import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import numpy as np

# Update to include the correct path
data_directory = "training/data"  # Path to the 'data' directory within 'training'
recording_name = os.path.join(data_directory, "FB-7B-DF-44-C3-1A.csv")  # Full file path
skip_rows = 0

# Create the figure for plotting
fig, (accelerometer_fig, gyroscope_fig) = plt.subplots(2, 1, figsize=(16, 10))

def animate(i):
    global skip_rows, recording_name
    if not os.path.exists(recording_name):
        print(f"File not found: {recording_name}")
        return
     # Read the CSV file into a DataFrame
    df = pd.read_csv(
        recording_name,
        header=None,
        names=["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        index_col=0,
        parse_dates=True,
        skiprows=skip_rows
    )
     # Adjust skip_rows to only keep the last 600 rows
    if len(df) >= 600:
        skip_rows += len(df) - 600
     # Clear the figures for updating
    accelerometer_fig.clear()
    gyroscope_fig.clear()

    # Plot Accelerometer Data
    accelerometer_fig.plot(df["acc_x"], color="red", label="X")
    accelerometer_fig.plot(df["acc_y"], color="green", label="Y")
    accelerometer_fig.plot(df["acc_z"], color="blue", label="Z")
    accelerometer_fig.set_yticks(np.arange(-2, 2.5, 0.5))
    accelerometer_fig.set_ylabel("Magnitude")
    accelerometer_fig.set_xlabel("Time")
    accelerometer_fig.set_title("Accelerometer Data")
    accelerometer_fig.legend()
    # Plot Gyroscope Data
    gyroscope_fig.plot(df["gyro_x"], color="cyan", label="X")
    gyroscope_fig.plot(df["gyro_y"], color="magenta", label="Y")
    gyroscope_fig.plot(df["gyro_z"], color="yellow", label="Z")
    gyroscope_fig.set_xlabel("Time")
    gyroscope_fig.set_yticks(np.arange(-1200, 1800, 600))
    gyroscope_fig.set_ylabel("Magnitude")
    gyroscope_fig.set_title("Gyroscope Data")
    gyroscope_fig.legend()
     # Update the title and adjust layout
    fig.suptitle(f"Wearable Data", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, top=0.88)

def live_plotting():
    # Set up animation to continuously update the plots
    ani = FuncAnimation(fig, animate, interval=60, cache_frame_data=False)
    plt.show()
