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

        # Odometry offset variables
        self._x_offset = 0.0
        self._y_offset = 0.0
        self._theta_offset = 0.0

        # Safety limits
        self.max_base_vel_xy = 0.15
        self.max_base_vel_theta = np.deg2rad(60.0)
        self.max_lift_vel = 0.2
        self.max_arm_vel = 0.1
        self.max_wrist_yaw_vel = np.deg2rad(60.0)
        self.max_wrist_pitch_vel = np.deg2rad(60.0)
        self.max_wrist_roll_vel = np.deg2rad(60.0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def reset_odometry_offset(self):
        """
        Resets the odometry offset to the current raw position of the robot.
        """
        self.robot.pull_status()
        status = self.robot.status
        base_status = status.get('omnibase', status.get('base', {}))
        self._x_offset = base_status.get('x', 0.0)
        self._y_offset = base_status.get('y', 0.0)
        self._theta_offset = base_status.get('theta', 0.0)

    def get_base_odometry(self, robot_status: dict) -> tuple[float, float, float]:
        """
        Extracts the base odometry from the robot status dictionary.
        """
        base_status = robot_status.get('omnibase', robot_status.get('base', {}))
        raw_x = base_status.get('x', 0.0)
        raw_y = base_status.get('y', 0.0)
        raw_theta = base_status.get('theta', 0.0)

        # 1. Translate relative to offset origin
        dx = raw_x - self._x_offset
        dy = raw_y - self._y_offset
        base_theta = raw_theta - self._theta_offset

        # 2. Rotate the relative translation to align with the reset orientation
        cos_t = np.cos(-self._theta_offset)
        sin_t = np.sin(-self._theta_offset)
        base_x = dx * cos_t - dy * sin_t
        base_y = dx * sin_t + dy * cos_t

        return base_x, base_y, base_theta

    def startup(self) -> bool:
        """
        Starts up the robot client connection if not already connected.
        """
        if getattr(self.robot, "server_connected", False) or getattr(self.robot, "is_valid", False):
            self.robot.pull_status()
            return True
        success = self.robot.startup()
        if success:
            self.robot.pull_status()
        return success

    def shutdown(self):
        """
        Safely stops the robot client connection.
        """
        self.cmd_zero_velocity()
        self.robot.stop()

    def get_joint_position(self) -> StretchJointPositions:
        """
        Queries the current robot status and constructs a StretchJointPositions object,
        pulling the base x, y, theta coordinates from odometry relative to the reset origin.
        """
        self.robot.pull_status()
        status = self.robot.status

        # Read base odometry (supporting 'omnibase' or legacy 'base' keys)
        base_x, base_y, base_theta = self.get_base_odometry(status)

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

    def cmd_velocities(self, v: StretchJointVelocities) -> None:
        """
        Commands velocities to all joints of the robot (base, lift, arm, and wrist joints).
        Enforces safety limits by clipping velocities and only commands homed joints.
        
        Args:
            v (StretchJointVelocities): The commanded velocities.
        """
        # Clip values to ensure safety
        base_x = np.clip(
            v.base_x,
            -self.max_base_vel_xy,
            self.max_base_vel_xy
        )
        base_y = np.clip(
            v.base_y,
            -self.max_base_vel_xy,
            self.max_base_vel_xy
        )
        base_theta = np.clip(
            v.base_theta,
            -self.max_base_vel_theta,
            self.max_base_vel_theta
        )
        
        lift = np.clip(
            v.lift,
            -self.max_lift_vel,
            self.max_lift_vel
        )
        arm = np.clip(
            v.arm,
            -self.max_arm_vel,
            self.max_arm_vel
        )
        
        wrist_yaw = np.clip(
            v.wrist_yaw,
            -self.max_wrist_yaw_vel,
            self.max_wrist_yaw_vel
        )
        wrist_pitch = np.clip(
            v.wrist_pitch,
            -self.max_wrist_pitch_vel,
            self.max_wrist_pitch_vel
        )
        wrist_roll = np.clip(
            v.wrist_roll,
            -self.max_wrist_roll_vel,
            self.max_wrist_roll_vel
        )

        # Base Velocity Command
        if hasattr(self.robot, "base"):
            try:
                self.robot.base.set_velocity(base_x, base_y, base_theta)
            except Exception as e:
                self.robot.logger.error(f"Failed to set base velocity: {e}")

        # Lift Velocity Command (Requires Homing)
        if hasattr(self.robot, "lift"):
            try:
                if self.robot.lift.is_homed():
                    self.robot.lift.set_velocity(lift)
                else:
                    self.robot.logger.warning("Lift joint is not homed; velocity command ignored.")
            except Exception as e:
                self.robot.logger.error(f"Failed to set lift velocity: {e}")

        # Arm Velocity Command (Requires Homing)
        if hasattr(self.robot, "arm"):
            try:
                if self.robot.arm.is_homed():
                    self.robot.arm.set_velocity(arm)
                else:
                    self.robot.logger.warning("Arm joint is not homed; velocity command ignored.")
            except Exception as e:
                self.robot.logger.error(f"Failed to set arm velocity: {e}")

        # End of Arm / Wrist Joints Velocity Command (Requires Homing per joint)
        if hasattr(self.robot, "end_of_arm"):
            for joint_name, val in [("wrist_yaw", wrist_yaw), ("wrist_pitch", wrist_pitch), ("wrist_roll", wrist_roll)]:
                try:
                    if joint_name in self.robot.end_of_arm.joints:
                        if self.robot.end_of_arm.is_homed(joint_name):
                            self.robot.end_of_arm.move_by(joint_name, val)  # TODO: replace with vel control later
                        else:
                            self.robot.logger.warning(f"Wrist joint {joint_name} is not homed; velocity command ignored.")
                except Exception as e:
                    self.robot.logger.error(f"Failed to set velocity for {joint_name}: {e}")

        # Push all queued commands to the hardware/server
        try:
            self.robot.push_command()
        except Exception as e:
            self.robot.logger.error(f"Failed to push velocity commands to the robot: {e}")

    def cmd_zero_velocity(self) -> None:
        self.cmd_velocities(
            StretchJointVelocities(
                base_x=0.0,
                base_y=0.0,
                base_theta=0.0,
                lift=0.0,
                arm=0.0,
                wrist_yaw=0.0,
                wrist_pitch=0.0,
                wrist_roll=0.0
            )
        )
