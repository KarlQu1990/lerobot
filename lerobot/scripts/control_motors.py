from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus


motor_name = "gripper"
motor_index = 4
motor_model = "sts3215"

motors_bus = FeetechMotorsBus(
        port="/dev/ttyCH341USB2",
        motors={"": (motor_index, motor_model)},
    )
motors_bus.connect()

position = motors_bus.read("Present_Position")

# move from a few motor steps as an example
few_steps = 50
motors_bus.write("Goal_Position", position + few_steps)

# when done, consider disconnecting
motors_bus.disconnect()