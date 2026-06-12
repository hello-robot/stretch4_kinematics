import numpy as np
from stretch4_body.robot.robot_client import RobotClient
from stretch4_kinematics.kinematic_models import (
    StretchJointPositions,
    StretchJointVelocities,
)

class StretchInterface:
    """
    Interface wrapper around RobotClient to convert raw hardware status
    dictionaries into kinematic representations (StretchJointPositions and StretchJointVelocities).
    """
    def __init__(self):
        self.robot = RobotClient()
        self.robot.startup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def startup(self) -> bool:
        """
        Starts up the robot client connection and pulls the initial status.
        """
        success = self.robot.startup()
        if success:
            self.robot.pull_status()
        return success

    def shutdown(self):
        """
        Safely stops the robot client connection.
        """
        self.robot.stop()

    def get_joint_position(self) -> StretchJointPositions:
        """
        Queries the current robot status and constructs a StretchJointPositions object,
        pulling the base x, y, theta coordinates from odometry.
        """
        self.robot.pull_status()
        status = self.robot.status

        # Read base odometry (supporting 'omnibase' or legacy 'base' keys)
        base_status = status.get('omnibase', status.get('base', {}))
        base_x = base_status.get('x', 0.0)
        base_y = base_status.get('y', 0.0)
        base_theta = base_status.get('theta', 0.0)

        # Read arm & lift
        lift_pos = status.get('lift', {}).get('pos', 0.5)
        arm_pos = status.get('arm', {}).get('pos', 0.0)

        # Read wrist joints
        eoa_status = status.get('end_of_arm', {})
        wrist_yaw = eoa_status.get('wrist_yaw', {}).get('pos', 0.0)
        wrist_pitch = eoa_status.get('wrist_pitch', {}).get('pos', 0.0)
        wrist_roll = eoa_status.get('wrist_roll', {}).get('pos', 0.0)

        return StretchJointPositions(
            base_x=base_x,
            base_y=base_y,
            base_theta=base_theta,
            lift=lift_pos,
            arm=arm_pos,
            wrist_yaw=wrist_yaw,
            wrist_pitch=wrist_pitch,
            wrist_roll=wrist_roll
        )

    def get_joint_velocity(self) -> StretchJointVelocities:
        """
        Queries the current robot status and constructs a StretchJointVelocities object,
        pulling the base x_vel, y_vel, theta_vel coordinates from odometry.
        """
        self.robot.pull_status()
        status = self.robot.status

        # Read base velocities from odometry
        base_status = status.get('omnibase', status.get('base', {}))
        base_x_vel = base_status.get('x_vel', 0.0)
        base_y_vel = base_status.get('y_vel', 0.0)
        base_theta_vel = base_status.get('theta_vel', 0.0)

        # Read arm & lift velocities
        lift_vel = status.get('lift', {}).get('vel', 0.0)
        arm_vel = status.get('arm', {}).get('vel', 0.0)

        # Read wrist velocities
        eoa_status = status.get('end_of_arm', {})
        wrist_yaw_vel = eoa_status.get('wrist_yaw', {}).get('vel', 0.0)
        wrist_pitch_vel = eoa_status.get('wrist_pitch', {}).get('vel', 0.0)
        wrist_roll_vel = eoa_status.get('wrist_roll', {}).get('vel', 0.0)

        return StretchJointVelocities(
            base_x=base_x_vel,
            base_y=base_y_vel,
            base_theta=base_theta_vel,
            lift=lift_vel,
            arm=arm_vel,
            wrist_yaw=wrist_yaw_vel,
            wrist_pitch=wrist_pitch_vel,
            wrist_roll=wrist_roll_vel
        )
