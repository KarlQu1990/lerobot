"""
This script configure a single motor at a time to a given ID and baudrate.

Example of usage:
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem585A0080521 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```
"""

import argparse


def show_motor(port, brand, model):
    if brand == "feetech":
        from lerobot.common.robot_devices.motors.feetech import MODEL_BAUDRATE_TABLE

        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus as MotorsBusClass
    elif brand == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import MODEL_BAUDRATE_TABLE
        from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus as MotorsBusClass
    else:
        raise ValueError(
            f"Currently we do not support this motor brand: {brand}. We currently support feetech and dynamixel motors."
        )

    # Check if the provided model exists in the model_baud_rate_table
    if model not in MODEL_BAUDRATE_TABLE:
        raise ValueError(
            f"Invalid model '{model}' for brand '{brand}'. Supported models: {list(MODEL_BAUDRATE_TABLE.keys())}"
        )

    motor_bus = MotorsBusClass(port=port, motors={"motor": (0, "sts3215")})

    # Try to connect to the motor bus and handle any connection-specific errors
    try:
        motor_bus.connect()
        print(f"Connected on port {motor_bus.port}")
    except OSError as e:
        print(f"Error occurred when connecting to the motor bus: {e}")
        return

    # Motor bus is connected, proceed with the rest of the operations
    try:
        ids = motor_bus.find_motor_indices([1, 2, 3, 4, 5, 6])
        print("ids:", ids)

    except Exception as e:
        print(f"Error occurred during motor configuration: {e}")

    finally:
        motor_bus.disconnect()
        print("Disconnected from motor bus.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True, help="Motors bus port (e.g. dynamixel,feetech)")
    parser.add_argument("--brand", type=str, required=True, help="Motor brand (e.g. dynamixel,feetech)")
    parser.add_argument("--model", type=str, required=True, help="Motor model (e.g. xl330-m077,sts3215)")
    args = parser.parse_args()

    show_motor(args.port, args.brand, args.model)
