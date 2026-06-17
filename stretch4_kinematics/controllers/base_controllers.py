import copy
import numpy as np

from stretch4_kinematics.state.joint_positions import StretchJointPositions
from stretch4_kinematics.state.joint_velocities import StretchJointVelocities

class StretchVelocityController:
    """
    Base velocity controller class for the Stretch 4 robot.
    Provides utilities for enforcing joint velocity and position limits.
    """
    def __init__(self):
        """
        Initializes the StretchVelocityController base class, defining maximum 
        joint velocities for safety clamping.
        """
        # joint velocity limits
        self._v_max = StretchJointVelocities(
            base_x=0.15,
            base_y=0.15,
            base_theta=np.deg2rad(60),
            lift=0.1,
            arm=0.1,
            wrist_yaw=np.deg2rad(60),
            wrist_pitch=np.deg2rad(60),
            wrist_roll=np.deg2rad(60)
        )

    def _enforce_joint_velocity_limits(
        self,
        v_desired: StretchJointVelocities,
    ) -> StretchJointVelocities:
        """
        Enforces joint velocity limits on the desired velocity commands.
        
        Args:
            v_desired (StretchJointVelocities): The desired velocity commands.
            
        Returns:
            StretchJointVelocities: The limited velocity commands.
        """
        v_limited = copy.deepcopy(v_desired)

        # clip each joint
        v_limited.base_x = np.clip(
            v_limited.base_x,
            -self._v_max.base_x,
            self._v_max.base_x
        )
        v_limited.base_y = np.clip(
            v_limited.base_y,
            -self._v_max.base_y,
            self._v_max.base_y
        )
        v_limited.base_theta = np.clip(
            v_limited.base_theta,
            -self._v_max.base_theta,
            self._v_max.base_theta
        )
        v_limited.lift = np.clip(
            v_limited.lift,
            -self._v_max.lift,
            self._v_max.lift
        )
        v_limited.arm = np.clip(
            v_limited.arm,
            -self._v_max.arm,
            self._v_max.arm
        )
        v_limited.wrist_yaw = np.clip(
            v_limited.wrist_yaw,
            -self._v_max.wrist_yaw,
            self._v_max.wrist_yaw
        )
        v_limited.wrist_pitch = np.clip(
            v_limited.wrist_pitch,
            -self._v_max.wrist_pitch,
            self._v_max.wrist_pitch
        )
        v_limited.wrist_roll = np.clip(
            v_limited.wrist_roll,
            -self._v_max.wrist_roll,
            self._v_max.wrist_roll
        )
        return v_limited

    def _enforce_joint_position_limits(self, q_desired: StretchJointPositions) -> StretchJointPositions:
        """
        Enforces joint position limits on the desired joint positions.
        Currently a placeholder that returns the input positions unchanged.

        Args:
             q_desired (StretchJointPositions): The desired joint positions.

        Returns:
             StretchJointPositions: The joint positions after enforcing limits.
        """
        q_limited = copy.deepcopy(q_desired)
        # TODO: Implement limits
        return q_limited

    def update(
        self,
        dt: float,
        current_pos: StretchJointPositions,
        current_vel: StretchJointVelocities
    ):
        """
        Abstract method to update the controller. Must be implemented by subclasses.

        Args:
            dt (float): Time step since last update.
            current_pos (StretchJointPositions): Current joint positions.
            current_vel (StretchJointVelocities): Current joint velocities.
        """
        raise NotImplementedError()
