from lerobot.scripts.display_cameras import main

if __name__ == "__main__":
    fps = 30
    pixel_format = "MJPG"
    color_mode = "bgr"

    main(fps, pixel_format, color_mode)
