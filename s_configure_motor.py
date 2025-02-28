from lerobot.scripts.configure_motor import configure_motor
from lerobot.common.utils.usb_utils import USBDeviceManager

if __name__ == "__main__":
    brand = "feetech"
    model = "sts3215"
    baudrate = 1000000
    port = "left_follower"
    ID = 6

    # 加载USB信息
    USBDeviceManager().load()

    configure_motor(port, brand, model, ID, baudrate)
