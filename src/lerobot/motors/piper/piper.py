import contextlib
import enum
import socket
import struct
import time
import traceback
from copy import deepcopy
from threading import Event, RLock, Thread
from typing import Any, List

import numpy as np

from lerobot.common.robot_devices.motors.configs import PiperMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

PIPER_CONTROL_TABLE = {
    "Present_Position": "joints",
    "Goal_Position": "joints",
    "Torque_Enable": "torque",
    "Torque_Disable": "torque",
}


MODEL_CONTROL_TABLE = {
    "piper": PIPER_CONTROL_TABLE,
}


class UDPSocketManager(object):
    _lock = RLock()
    _instance = None
    _first_init = False

    port = 5010

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with UDPSocketManager._lock:
                cls._instance = super().__new__(cls, *args, **kwargs)

        return cls._instance

    def __init__(self):
        if self._first_init:
            return

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("0.0.0.0", self.port))
            self._first_init = True
        except Exception:
            raise

    def get(self) -> socket.socket:
        return self._sock

    def close(self):
        with contextlib.suppress(Exception):
            self._sock.close()


class ControlMode(enum.Enum):
    disable = 0  # 禁止任何操作
    follower_control = 1  # 从臂控制
    teleoperate = 2  # 主臂遥操作从臂


class ArmUDPController(object):
    local_port = 5010  # 本机服务端口
    remote_host = "192.168.191.60"

    def __init__(self, remote_host: str, remote_port: int, is_leader: bool = True):
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.is_leader = is_leader
        self.frame_id = 0

        self._stop_event = Event()
        self._recv_data = {}
        self._lock = RLock()

        self._mode = ControlMode.disable
        self._enable = True
        self._emergency = False

    def switch_mode(self, mode: ControlMode):
        with self._lock:
            self._mode = mode

    def start(self):
        self.recv_thread = Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

    def write(self, query_name: str, values: Any):
        if query_name == "joints":
            self._write_joints(values)

    def _write_joints(self, joints: List[float]):
        emergency_flag = 0x01 if self._emergency else 0x00
        enable_flag = 0x01 if self._enable else 0x00

        data = struct.pack("<BBBB", 0x55, 0x01, emergency_flag, enable_flag)
        for j in joints:
            data += struct.pack("<i", int(j * 1000))

        padding_len = 64 - len(data) - 5
        data += b"\x00" * padding_len

        mode_flag = self._mode.value

        data += struct.pack("<B", mode_flag)
        data += struct.pack("<HB", 101, self.frame_id % 256)
        data += struct.pack("<B", 0x0A)

        addr = (self.remote_host, self.remote_port)

        sock = UDPSocketManager().get()
        sock.sendto(data, addr)
        self.frame_id += 1

    def _recv_loop(self):
        sock = UDPSocketManager().get()
        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(128)
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
            except Exception as e:
                print("接收错误:", e)

    def stop(self):
        self._stop_event.set()
        UDPSocketManager().close()
        print("Socket已关闭。")

    def read(self, query_name: str):
        if query_name == "joints":
            return self._read_joints()

    def _read_joints(self):
        prefix = "leader" if self.is_leader else "follower"
        return self._recv_data.get(f"{prefix}_joints")


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


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class PiperMotorsBus(object):
    def __init__(
        self,
        config: PiperMotorsBusConfig,
    ):
        self.port = config.port
        self.host = config.host
        self.motors = config.motors
        self.is_leader = config.is_leader

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)

        self.calibration = None
        self.is_connected = False

        self.logs = {}
        self.track_positions = {}

        self.controller = ArmUDPController(self.host, self.port, is_leader=self.is_leader)

    def to_teleoperate_mode(self):
        """切换为遥操作模式。"""
        self.controller.switch_mode(ControlMode.teleoperate)

    def to_follower_control_mode(self):
        """切换为从臂控制模式（模型驱动）。"""
        self.controller.switch_mode(ControlMode.follower_control)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"{self.__class__.__name__}({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        try:
            self.controller.start()
            query_name = self.data_name_to_query_name("Present_Position")
            data = self.controller.read(query_name)
            retry_cnt = 10
            cnt = 0
            while not data and cnt < retry_cnt:
                time.sleep(0.5)
                data = self.controller.read(query_name)
                cnt += 1

            if not data:
                raise OSError(f"Connection failed: ({self.host}, {self.port})")
        except Exception:
            traceback.print_exc()
            raise

        # Allow to read and write
        self.is_connected = True

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def data_name_to_query_name(self, data_name: str):
        return self.model_ctrl_table[self.motor_models[0]][data_name]

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

        query_name = self.data_name_to_query_name(data_name)
        joints = self.controller.read(query_name)
        if not joints:
            raise ConnectionError("Read failed.")

        for idx in motor_ids:
            value = joints[idx - 1]
            values.append(value)

        values = np.array(values, dtype=np.float32)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        start_time = time.perf_counter()

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
        query_name = self.data_name_to_query_name(data_name)
        self.controller.write(query_name, values)

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def disconnect(self):
        if not self.is_connected:
            return

        self.controller.switch_mode(ControlMode.disable)
        self.controller.stop()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
