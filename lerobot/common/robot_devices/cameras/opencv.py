"""
This file contains utilities for recording frames from cameras. For more info look at `OpenCVCamera` docstring.
"""
import cv2
import argparse
import concurrent.futures
import math
import platform
import shutil
import threading
import time
import subprocess
import multiprocessing as mp
from multiprocessing.synchronize import Event as MpEvent
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Thread


import numpy as np
from PIL import Image

from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc

# The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60


cv2.setNumThreads(1)


def find_cameras(raise_when_empty=False, max_index_search_range=MAX_OPENCV_INDEX, mock=False) -> list[dict]:
    cameras = []
    if platform.system() == "Linux":
        print("Linux detected. Finding available camera indices through scanning '/dev/video*' ports")
        possible_ports = [str(port) for port in Path("/dev").glob("video*")]
        ports = _find_cameras_linux(possible_ports, mock=mock)
        for port in ports:
            if Path(port).is_symlink():
                continue

            cameras.append(
                {
                    "port": port,
                    "index": int(port.removeprefix("/dev/video")),
                }
            )
    else:
        print(
            "Mac or Windows detected. Finding available camera indices through "
            f"scanning all indices from 0 to {MAX_OPENCV_INDEX}"
        )
        possible_indices = range(max_index_search_range)
        indices = _find_cameras(possible_indices, mock=mock)
        for index in indices:
            cameras.append(
                {
                    "port": None,
                    "index": index,
                }
            )

    return cameras


def _find_cameras_linux(possible_ports: list[int | str], raise_when_empty=False, mock=False
) -> list[int | str]:
    if mock:
        return _find_cameras(possible_ports, raise_when_empty=raise_when_empty, mock=True)

    ports = []
    for port in possible_ports:
        output = subprocess.run(f"v4l2-ctl -d {port} --list-formats-ext", shell=True, capture_output=True)
        if output.stderr or "Size" not in output.stdout.decode():
            continue

        ports.append(port)


    if raise_when_empty and len(ports) == 0:
        raise OSError("未检测到摄像头，请重新插拔USB摄像头。")

    return ports


def _find_cameras(
    possible_camera_ids: list[int | str], raise_when_empty=False, mock=False
) -> list[int | str]:
    if mock:
        import tests.mock_cv2 as cv2
    else:
        import cv2

    camera_ids = []
    for camera_idx in possible_camera_ids:
        camera = cv2.VideoCapture(camera_idx)
        is_open = camera.isOpened()
        camera.release()

        if is_open:
            print(f"Camera found at index {camera_idx}")
            camera_ids.append(camera_idx)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `opencv2`, "
            "or your camera driver, or make sure your camera is compatible with opencv2."
        )

    return camera_ids


def is_valid_unix_path(path: str) -> bool:
    """Note: if 'path' points to a symlink, this will return True only if the target exists"""
    p = Path(path)
    return p.is_absolute() and p.exists()


def get_camera_index_from_unix_port(port: Path) -> int:
    if port.is_symlink():
        port = Path("/dev") / port.readlink()

    return int(str(port.resolve()).removeprefix("/dev/video"))


def save_image(img_array, camera_index, frame_index, images_dir):
    img = Image.fromarray(img_array)
    path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_cameras(
    images_dir: Path,
    camera_ids: list | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    mock=False,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given camera index.
    """
    if camera_ids is None or len(camera_ids) == 0:
        camera_infos = find_cameras(mock=mock)
        camera_ids = [cam["index"] for cam in camera_infos]

    print("Connecting cameras")
    cameras = []
    for cam_idx in camera_ids:
        camera = OpenCVCamera(cam_idx, fps=fps, width=width, height=height, mock=mock)
        camera.connect()
        print(
            f"OpenCVCamera({camera.camera_index}, fps={camera.fps}, width={camera.width}, "
            f"height={camera.height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            now = time.perf_counter()

            for camera in cameras:
                # If we use async_read when fps is None, the loop will go full speed, and we will endup
                # saving the same images from the cameras multiple times until the RAM/disk is full.
                image = camera.read() if fps is None else camera.async_read()

                executor.submit(
                    save_image,
                    image,
                    camera.camera_index,
                    frame_index,
                    images_dir,
                )

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break

            frame_index += 1

    print(f"Images have been saved to {images_dir}")


@dataclass
class OpenCVCameraConfig:
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(30, 640, 480)
    OpenCVCameraConfig(60, 640, 480)
    OpenCVCameraConfig(90, 640, 480)
    OpenCVCameraConfig(30, 1280, 720)
    ```
    """

    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    rotation: int | None = None
    mock: bool = False
    pixel_format: str = "YUYV"

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")

        if self.pixel_format not in ["YUYV", "MJPG"]:
            raise ValueError(f"`pixel_format` must be in ['YUYV', 'MJPG'] (got {self.rotation})")
        

class CameraProcess:
    def __init__(self, camera_idx: int, fps: int, width: int, height: int, color_mode: str, pixel_format: str, mock: bool, rotation: int ):
        self.camera_index = camera_idx
        self.fps = fps
        self.width = width
        self.height = height
        self.color_mode = color_mode
        self.mock = mock
        self.pixel_format = pixel_format

        self.is_connected = False
        self.color_image = None
        self.rotation = rotation
        self.logs = {}

        self.camera = None
        self.queue = None

        self.stop_event = None
        self.read_process = None
        self.read_thread = None

    def read(self, temporary_color_mode: str | None = None, camera: "cv2.VideoCapture" = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )
        
        if camera is None and self.stop_event is not None:
            raise Exception(
                f"OpenCVCamera({self.camera_index}) already run in multiprocess mode, can not switch mode, please use `async_read`."
            )
        
        if camera is None:
            camera = self.camera

        start_time = time.perf_counter()

        ret, color_image = camera.read()

        if not ret:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")

        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode

        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == "rgb":
            if self.mock:
                import tests.mock_cv2 as cv2
            else:
                import cv2

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        if h != self.height or w != self.width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        # log the number of seconds it took to read the image
        delta_timestamp_s = time.perf_counter() - start_time

        # log the utc time at which the image was received
        timestamp_utc = capture_timestamp_utc()


        return delta_timestamp_s, timestamp_utc, color_image

    def read_process_loop(self, queue: mp.Queue, stop_event: MpEvent):
        if self.camera:
            self.camera.release()
            self.camera = None
            time.sleep(0.1)

        camera = self._create_video_capture()

        while not stop_event.is_set():
            try:
                result = self.read(camera=camera)
                queue.put(result)
            except Exception as e:
                print(f"Error reading in process: {e}")

    def read_thread_loop(self, queue: mp.Queue, stop_event: MpEvent):
        while not stop_event.is_set():
            if not queue.empty():
                result = queue.get()
                if not result:
                    break

                delta_timestamp_s, timestamp_utc, color_image = result
                self.logs["delta_timestamp_s"] = delta_timestamp_s
                self.logs["timestamp_utc"] = timestamp_utc
                self.color_image = color_image

    def asyn_read(self):
        if self.read_process is None:
            self.stop_event = mp.Event()
            self.queue = mp.Queue()
            self.read_process = Thread(target=self.read_process_loop, args=(self.queue, self.stop_event))
            self.read_process.start()

            self.read_thread = Thread(target=self.read_thread_loop, args=(self.queue, self.stop_event))
            self.read_thread.daemon = True
            self.read_thread.start()

        num_tries = 0
        while True:
            if self.color_image is not None:
                return self.color_image

            time.sleep(1 / self.fps)
            num_tries += 1
            if num_tries > self.fps * 2:
                raise TimeoutError("Timed out waiting for async_read() to start.")

    def _create_video_capture(self):
        camera_idx = f"/dev/video{self.camera_index}" if platform.system() == "Linux" else self.camera_index

        camera = cv2.VideoCapture(camera_idx, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.pixel_format))

        if self.fps is not None:
            camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.width is not None:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
        if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            # Using `OSError` since it's a broad that encompasses issues related to device communication
            raise OSError(
                f"Can't set {self.fps=} for OpenCVCamera({self.camera_index}). Actual value is {actual_fps}."
            )
        if self.width is not None and not math.isclose(self.width, actual_width, rel_tol=1e-3):
            raise OSError(
                f"Can't set {self.width=} for OpenCVCamera({self.camera_index}). Actual value is {actual_width}."
            )
        if self.height is not None and not math.isclose(self.height, actual_height, rel_tol=1e-3):
            raise OSError(
                f"Can't set {self.height=} for OpenCVCamera({self.camera_index}). Actual value is {actual_height}."
            )
    
        self.fps = round(actual_fps)
        self.width = round(actual_width)
        self.height = round(actual_height)

        return camera


    def connect(self):
        if self.mock:
            import tests.mock_cv2 as cv2
        else:
            import cv2

            # Use 1 thread to avoid blocking the main thread. Especially useful during data collection
            # when other threads are used to save the images.
            cv2.setNumThreads(1)

        camera_idx = f"/dev/video{self.camera_index}" if platform.system() == "Linux" else self.camera_index
        # First create a temporary camera trying to access `camera_index`,
        # and verify it is a valid camera by calling `isOpened`.
        tmp_camera = cv2.VideoCapture(camera_idx)
        is_camera_open = tmp_camera.isOpened()
        # Release camera to make it accessible for `find_camera_indices`
        tmp_camera.release()
        del tmp_camera

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `camera_index` is valid before printing the traceback
            cameras_info = find_cameras()
            available_cam_ids = [cam["index"] for cam in cameras_info]
            if self.camera_index not in available_cam_ids:
                raise ValueError(
                    f"`camera_index` is expected to be one of these available cameras {available_cam_ids}, but {self.camera_index} is provided instead. "
                    "To find the camera index you should use, run `python lerobot/common/robot_devices/cameras/opencv.py`."
                )

            raise OSError(f"Can't access OpenCVCamera({camera_idx}).")

        # Secondly, create the camera that will be used downstream.
        # Note: For some unknown reason, calling `isOpened` blocks the camera which then
        # needs to be re-created.
        self.camera = self._create_video_capture()

        self.is_connected = True

    def disconnect(self):
        if self.read_process is not None:
            self.stop_event.set()
            self.queue.put(None)

            self.read_process.join()
            self.read_thread.join()

            self.read_process = None
            self.read_thread = None
            self.stop_event = None

        if self.camera:
            self.camera.release()
            self.camera = None


class OpenCVCamera: 
    """
    The OpenCVCamera class allows to efficiently record images from cameras. It relies on opencv2 to communicate
    with the cameras. Most cameras are compatible. For more info, see the [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

    An OpenCVCamera instance requires a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera
    like a webcam of a laptop, the camera index is expected to be 0, but it might also be very different, and the camera index
    might change if you reboot your computer or re-plug your camera. This behavior depends on your operation system.

    To find the camera indices of your cameras, you can run our utility script that will be save a few frames for each camera:
    ```bash
    python lerobot/common/robot_devices/cameras/opencv.py --images-dir outputs/images_from_opencv_cameras
    ```

    When an OpenCVCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of usage:
    ```python
    camera = OpenCVCamera(camera_index=0)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    camera = OpenCVCamera(0, fps=30, width=1280, height=720)
    camera = connect()  # applies the settings, might error out if these settings are not compatible with the camera

    camera = OpenCVCamera(0, fps=90, width=640, height=480)
    camera = connect()

    camera = OpenCVCamera(0, fps=90, width=640, height=480, color_mode="bgr")
    camera = connect()
    ```
    """

    def __init__(self, camera_index: int | str, config: OpenCVCameraConfig | None = None, **kwargs):
        if config is None:
            config = OpenCVCameraConfig()

        # Overwrite config arguments using kwargs
        config = replace(config, **kwargs)

        self.camera_index = camera_index
        self.port = None

        # Linux uses ports for connecting to cameras
        if platform.system() == "Linux":
            if isinstance(self.camera_index, int):
                self.port = Path(f"/dev/video{self.camera_index}")
            elif isinstance(self.camera_index, str) and is_valid_unix_path(self.camera_index):
                self.port = Path(self.camera_index)
                # Retrieve the camera index from a potentially symlinked path
                self.camera_index = get_camera_index_from_unix_port(self.port)
            else:
                raise ValueError(f"Please check the provided camera_index: {camera_index}")
            
        if config.mock:
            import tests.mock_cv2 as cv2
        else:
            import cv2
        
        self.is_connected = False

        rotation = None
        if config.rotation == -90:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            rotation = cv2.ROTATE_180

        self.camera_process = CameraProcess(self.camera_index, config.fps, config.width, config.height, config.color_mode, config.pixel_format, config.mock, rotation)

    @property
    def logs(self):
        return self.camera_process.logs
    
    @property
    def color_image(self):
        return self.camera_process.color_image

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"OpenCVCamera({self.camera_index}) is already connected.")

        self.camera_process.connect()
        self.is_connected = True

    def read(self, temporary_color_mode: str | None = None, camera: "cv2.VideoCapture" = None) -> np.ndarray:
        """Read a frame from the camera returned in the format (height, width, channels)
        (e.g. 480 x 640 x 3), contrarily to the pytorch format which is channel first.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        delta_timestamp_s, timestamp_utc, color_image = self.camera_process.read(temporary_color_mode, camera=camera)

        # 在主进程中调用，需要手动更新时间辍信息
        self.camera_process.logs["delta_timestamp_s"] = delta_timestamp_s
        self.camera_process.logs["timestamp_utc"] = timestamp_utc
        self.camera_process.color_image = color_image

        return color_image

    def async_read(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        return self.camera_process.asyn_read()
    
    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        self.camera_process.disconnect()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `OpenCVCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of camera indices used to instantiate the `OpenCVCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default=None,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=str,
        default=None,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_opencv_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
