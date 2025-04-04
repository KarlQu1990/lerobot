import argparse
import logging
import math
import sys

import cv2
import numpy as np

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera, OpenCVCameraConfig, find_cameras

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def main(fps: str, pixel_format: str, color_mode: str):
    camera_infos = find_cameras()
    if not camera_infos:
        logger.info("未找到可用摄像头。")
        sys.exit()

    camera_num = len(camera_infos)
    cols = min(2, camera_num)
    rows = math.ceil(camera_num / cols)
    width = 640
    height = 480
    concat_img = np.zeros((height * rows, width * cols, 3), np.uint8)

    cameras = []
    logger.info(f"发现摄像头: 数量={camera_num}, 信息：{camera_infos}")
    for i, info in enumerate(camera_infos):
        row = i // cols
        col = i % cols

        idx = info["index"]

        cfg = OpenCVCameraConfig(
            idx, width=width, height=height, fps=fps, color_mode=color_mode, pixel_format=pixel_format
        )
        camera = OpenCVCamera(cfg)
        try:
            camera.connect()
        except Exception as e:
            logger.error(f"连接摄像头错误: {e}, 请检查摄像头{info['port']}")

        if not camera.is_connected:
            cameras.append([(row, col), None])
            continue

        # 打印摄像头详细信息
        # info = subprocess.check_output(f"v4l2-ctl -d /dev/video{idx} --list-formats-ext", shell=True)
        # logger.info(f"摄像头/dev/video{idx}分辨率和帧率信息:\n{info.decode()}")
        camera.async_read()
        cameras.append([(row, col), camera])

    try:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        while True:
            for (row, col), camera in cameras:
                if not camera or camera.color_image is None:
                    continue

                start_y = row * height
                end_y = (row + 1) * height
                start_x = col * width
                end_x = (col + 1) * width
                concat_img[start_y:end_y, start_x:end_x, :] = camera.color_image
                cv2.imshow("frame", concat_img)
                if cv2.waitKey(5) == ord("q"):
                    break

    except Exception:
        logger.exception("读取摄像头流异常::")
        for _, camera in cameras:
            if camera:
                camera.disconnect()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30, help="视频帧率。")
    parser.add_argument("--pixel-format", type=str, default="MJPG", choices=["YUYV", "MJPG"], help="像素格式。")
    parser.add_argument("--color-mode", type=str, default="bgr", choices=["bgr", "rgb"], help="颜色通道模式")
    args = parser.parse_args()

    main(args.fps, args.pixel_format, args.color_mode)
