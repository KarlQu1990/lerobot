import logging
import sys
import time
import os
import rerun as rr

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera, OpenCVCameraConfig, find_cameras
from lerobot.common.robot_devices.utils import busy_wait

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def init_rerun() -> None:
    # Configure Rerun flush batch size default to 8KB if not set
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

    # Initialize Rerun based on configuration
    rr.init("lerobot_display_cameras")
    # Get memory limit for rerun viewer parameters
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def main():
    camera_infos = find_cameras()
    if not camera_infos:
        logger.info("未找到可用摄像头。")
        sys.exit()

    fps = 30
    width = 640
    height = 480
    pixel_format = "MJPG"

    cameras = []
    logger.info(f"发现摄像头: 数量={len(camera_infos)}, 信息：{camera_infos}")
    for info in camera_infos:
        idx = info["index"]

        cfg = OpenCVCameraConfig(idx, width=width, height=height, fps=fps, pixel_format=pixel_format)
        camera = OpenCVCamera(cfg)
        try:
            camera.connect()
        except Exception as e:
            logger.exception(f"连接摄像头错误: {e}, 请检查摄像头{info['port']}")

        if not camera.is_connected:
            cameras.append(None)
            continue

        camera.async_read()
        cameras.append(camera)

    try:
        while True:
            start_t = time.perf_counter()
            for i, camera in enumerate(cameras):
                if not camera or camera.color_image is None:
                    continue

                rr.log(f"camera{i}", rr.Image(camera.color_image), static=True)

            dt_s = time.perf_counter() - start_t
            busy_wait(1 / cfg.fps - dt_s)

    except Exception:
        logger.exception("读取摄像头流异常::")
        for _, camera in cameras:
            if camera:
                camera.disconnect()


if __name__ == "__main__":
    init_rerun()
    main()
