from lerobot.teleoperators.sam01_leader import SAM01Leader, SAM01LeaderConfig
from lerobot.robots.sam01_follower import SAM01Follower, SAM01FollowerConfig

robot_config = SAM01FollowerConfig(
    port="left_follower",
    id="left_robot_arm",
)

teleop_config = SAM01LeaderConfig(
    port="left_follower",
    id="left_robot_arm",
)

robot = SAM01Follower(robot_config)
teleop_device = SAM01Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)