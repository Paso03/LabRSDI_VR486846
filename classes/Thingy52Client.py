import asyncio
import numpy as np
from bleak import BleakClient, BLEDevice
from datetime import datetime
import struct
from torchmetrics.functional.audio.dnsmos import ort
from utils.UUIDs import TMS_RAW_DATA_UUID, TMS_CONF_UUID
from utils.utility import motion_characteristics, change_status, get_uuid
import os
import onnxruntime as ort

class Thingy52Client(BleakClient):

    def __init__(self, device: BLEDevice):
        super().__init__(get_uuid(device))
        self.path = "training/data"  # Define the directory path where data should be saved
        self.mac_address = device.address

        self.model = ort.InferenceSession('training/CNN_60.onnx')
        self.classes = ["sleeping", "writing"]

        self.buffer_size = 60
        self.data_buffer = []

        self.recording_name = None
        self.file = None

        # Ensure that the directory exists
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    async def connect(self, **kwargs) -> bool:
        try:
            await super().connect(**kwargs)
            print(f"Successfully connected to {self.mac_address}")
            print(f"Device {self.mac_address} is now connected.")
            return True
        except Exception as e:
            print(f"Failed to connect to {self.address}: {e}")
            return False

    def disconnect(self) -> bool:
        print(f"Disconnecting from {self.mac_address}")
        if self.file:
            self.file.close()
        return super().disconnect()

    async def receive_inertial_data(self, sampling_frequency: int = 60):
        payload = motion_characteristics(motion_processing_unit_freq=sampling_frequency)
        await self.write_gatt_char(TMS_CONF_UUID, payload)

        # Open file in the data directory
        if self.recording_name:
            self.file = open(self.recording_name, "a+")
        else:
            print("No recording name provided!")
            return

        await self.start_notify(TMS_RAW_DATA_UUID, self.raw_data_callback)

        await change_status(self, "recording")

        try:
            while True:
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            await self.stop_notify(TMS_RAW_DATA_UUID)
            print("Stopped notification")

    def save_to(self, file_name):
        # Ensure the file is being saved in the correct directory
        self.recording_name = f"{self.mac_address.replace(':', '-')}_{file_name}.csv"

    def raw_data_callback(self, sender, data):
        receive_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Accelerometer
        acc_x = (struct.unpack('h', data[0:2])[0] * 1.0) / 2 ** 10
        acc_y = (struct.unpack('h', data[2:4])[0] * 1.0) / 2 ** 10
        acc_z = (struct.unpack('h', data[4:6])[0] * 1.0) / 2 ** 10

        # Gyroscope
        gyro_x = (struct.unpack('h', data[6:8])[0] * 1.0) / 2 ** 5
        gyro_y = (struct.unpack('h', data[8:10])[0] * 1.0) / 2 ** 5
        gyro_z = (struct.unpack('h', data[10:12])[0] * 1.0) / 2 ** 5

        # Compass
        comp_x = (struct.unpack('h', data[12:14])[0] * 1.0) / 2 ** 4
        comp_y = (struct.unpack('h', data[14:16])[0] * 1.0) / 2 ** 4
        comp_z = (struct.unpack('h', data[16:18])[0] * 1.0) / 2 ** 4

        # Write data to file
        if self.file:
            self.file.write(f"{receive_time},{acc_x},{acc_y},{acc_z},{gyro_x},{gyro_y},{gyro_z}\n")

        # Manage buffer size and add new data
        if len(self.data_buffer) == self.buffer_size:
            input_data = np.array(self.data_buffer, dtype=np.float32).reshape(1, self.buffer_size, 6)
            input_ = self.model.get_inputs()[0].name
            cls_index = np.argmax(self.model.run(None, {input_: input_data})[0], axis=1)[0]
            print(f"\r{self.mac_address} | {receive_time} - Prediction: {self.classes[cls_index]}", end="", flush=True)
            self.data_buffer.clear()

        self.data_buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])

        # print(f"\r{self.mac_address} | {receive_time} - Accelerometer: X={acc_x: 2.3f}, Y={acc_y: 2.3f}, Z={acc_z: 2.3f}", end="", flush=True)