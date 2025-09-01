from lerobot.teleoperators.bi_sam01_leader import BiSAM01Leader, BiSAM01LeaderConfig
from lerobot.robots.bi_sam01_follower import BiSAM01Follower, BiSAM01FollowerConfig

robot_config = BiSAM01FollowerConfig(
    left_arm_port="left_follower",
    right_arm_port="right_follower",
    id="robot_arm_follower",
)

teleop_config = BiSAM01LeaderConfig(

    left_arm_port="left_leader",
    right_arm_port="right_leader",
    id="robot_arm_leader",
)

robot = BiSAM01Follower(robot_config)
teleop_device = BiSAM01Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)