import contextlib
import logging
import socket
import struct
import time
import traceback
from threading import Event, RLock, Thread
from typing import List

import numpy as np

from lerobot.common.robot_devices.motors.configs import PiperMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

PIPER_CONTROL_TABLE = {
    "Present_Position": (3, 2),
    "Goal_Position": (5, 1),
}


MODEL_CONTROL_TABLE = {
    "piper": PIPER_CONTROL_TABLE,
}


class UDPSocketManager(object):
    _instance = None

    port = 5010

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UDPSocketManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("0.0.0.0", self.port))
        except Exception:
            raise

    def get(self) -> socket.socket:
        return self.sock

    def close(self):
        with contextlib.suppress(Exception):
            self.sock.close()


class ArmUDPController(object):
    local_port = 5010  # 本机服务端口
    remote_host = "192.168.191.60"

    def __init__(self, remote_host: str, remote_port: int):
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.frame_id = 0

        self._stop_event = Event()
        self._recv_data = {}
        self._lock = RLock()

    def start(self):
        self.recv_thread = Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def write(self, joints: List[float], emergency: bool = False, enable: bool = True, teleoperate_mode: bool = True):
        if len(joints) != 7:
            raise ValueError("需要提供7个关节角度")

        emergency_flag = 0x01 if emergency else 0x00
        enable_flag = 0x01 if enable else 0x00

        data = struct.pack("<BBBB", 0x55, 0x01, emergency_flag, enable_flag)
        for j in joints:
            data += struct.pack("<i", int(j * 1000))

        padding_len = 64 - len(data) - 5
        data += b"\x00" * padding_len

        mode_flag = 0x01 if teleoperate_mode else 0x00

        data += struct.pack("<B", mode_flag)
        data += struct.pack("<HB", 101, self.frame_id % 256)
        data += struct.pack("<B", 0x0A)

        addr = (self.remote_host, self.remote_port)
        logging.info(f"[发送] → {addr} 数据(HEX):", data.hex())

        sock = UDPSocketManager().get()
        sock.sendto(data, addr)
        self.frame_id += 1

    def _recv_loop(self):
        sock = UDPSocketManager().get()
        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(128)

                if len(data) != 128 or data[0] != 0xAA:
                    continue

                follower = [struct.unpack("<i", data[4 + i * 4 : 8 + i * 4])[0] / 1000.0 for i in range(7)]
                leader = [struct.unpack("<i", data[32 + i * 4 : 36 + i * 4])[0] / 1000.0 for i in range(7)]

                with self._lock:
                    self._recv_data = {
                        "timestamp": time.time(),
                        "emergency": data[2],
                        "enable": data[3],
                        "follower_joints": follower,
                        "leader_joints": leader,
                        "version": struct.unpack("<H", data[124:126])[0],
                        "frame_id": data[126],
                    }

                    logging.info(f"[接收] ← {addr} 状态帧 ID:{self._recv_data['frame_id']}")
            except Exception as e:
                logging.info("接收错误:", e)

    def stop(self):
        self._stop_event.set()
        UDPSocketManager().close()
        logging.info("[停止] Socket 已关闭。")

    def recover(self, teleoperate_mode: bool = False):
        logging.info("[恢复流程] 发送急停指令...")
        self.write([0] * 7, emergency=True, enable=False, teleoperate_mode=teleoperate_mode)
        time.sleep(1)
        logging.info("[恢复流程] 发送恢复启用指令...")
        self.write([0] * 7, emergency=False, enable=True, teleoperate_mode=teleoperate_mode)

    def read(self):
        return self._recv_data


def get_group_sync_key(data_name, motor_names):
    group_key = f"{data_name}_" + "_".join(motor_names)
    return group_key


def get_result_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    rslt_name = f"{fn_name}_{group_key}"
    return rslt_name


def get_queue_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    queue_name = f"{fn_name}_{group_key}"
    return queue_name


def get_log_name(var_name, fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name


class MotorsBus(object):
    def __init__(
        self,
        config: PiperMotorsBusConfig,
    ):
        self.port = config.port
        self.motors = config.motors
        self.mock = config.mock
        self.is_leader = config.is_leader

        # self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        # self.model_resolution = deepcopy(MODEL_RESOLUTION)

        self.port_handler = None
        self.packet_handler = None
        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}

        self.track_positions = {}

        self.controller = ArmUDPController(self.host, self.port)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"FeetechMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        try:
            self.controller.start()
            data = self.controller.read()
            retry_cnt = 10
            cnt = 0
            while not data and cnt < retry_cnt:
                time.sleep(0.5)
                data = self.controller.read()

        except Exception:
            traceback.print_exc()
            raise

        # Allow to read and write
        self.is_connected = True

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def read(self, data_name, motor_names: str | list[str] | None = None):
        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        values = []

        data = self.controller.read()
        prefix = "leader" if self.is_leader else "follower"
        joints = data[f"{prefix}_joints"]

        for idx in motor_ids:
            value = joints[idx - 1]
            values.append(value)

        values = np.array(values)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        start_time = time.perf_counter()

        if self.mock:
            pass
        else:
            pass

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        values = values.tolist()

        # TODO:写逻辑实现

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def disconnect(self):
        pass

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
