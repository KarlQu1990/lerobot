from pathlib import Path

import numpy as np
from roboticstoolbox.robot import Robot

ASSET_ROOT = Path(__file__).parents[3] / "assets"


class RobotKinematics(Robot):
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] = None,
    ):
        links, name, urdf_string, urdf_filepath = self.URDF_read(urdf_path)

        frame_id = -1
        for i, link in enumerate(links):
            if link.name == target_frame_name:
                frame_id = i
        if not frame_id:
            raise ValueError("target link {} not in list: {}", target_frame_name, [link.name for link in links])

        super().__init__(
            links[:-1],
            name=name.upper(),
            manufacturer="SAM",
            gripper_links=None,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
        self.ilimit = 50
        self.slimit = 100
        self.tol = 1e-5
        self.qlim = [[-3.14, -1.57, -1.57, -1.57, -1.57, -3.14], [3.14, 1.57, 1.57, 1.57, 1.57, 3.14]]

    def forward_kinematics(self, joint_pos_deg):
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        current_joint_rad = np.deg2rad(joint_pos_deg[:-1])
        mat = self.fkine(current_joint_rad)
        x, y, z = mat.t
        rotation_mat = mat.R  # 3x3 matrix
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rotation_mat
        trans_mat[:3, 3] = [x, y, z]

        return trans_mat

    def inverse_kinematics(self, current_joint_pos, desired_ee_pose):
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        current_joint_rad = np.deg2rad(current_joint_pos[:-1])

        sol = self.ikine_LM(
            Tep=desired_ee_pose,
            q0=current_joint_rad,
            ilimit=self.ilimit,
            slimit=self.slimit,
            tol=self.tol,
        )

        if sol.success:
            joint_pos_rad = sol.q
            joint_pos_deg = np.rad2deg(joint_pos_rad)
            return np.append(joint_pos_deg, current_joint_pos[-1])
        else:
            print("IK fails")
            return current_joint_pos
