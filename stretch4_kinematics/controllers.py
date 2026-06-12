import copy
from enum import Enum, auto
import numpy as np
import pinocchio as pin

from stretch4_kinematics.kinematic_models import (
    StretchJointPositions,
    StretchJointVelocities,
    ToolFrameKinematics
)


class FlyingGripperState(Enum):
    """
    Flying gripper controller states.
    """
    IDLE = auto()
    APPROACH = auto()
    RETRACT = auto()
    FINISH = auto()
    FAIL = auto()


class StretchVelocityController:
    """
    Base velocity controller class for the Stretch 4 robot.
    Provides utilities for enforcing joint velocity and position limits.
    """
    def __init__(self):
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
        q_limited = copy.deepcopy(q_desired)
        # TODO: Implement limits
        return q_limited

    def update(self, dt: float, current_pos: StretchJointPositions, current_vel: StretchJointVelocities):
        raise NotImplementedError()


class FlyingGripperController(StretchVelocityController):
    def __init__(self):
        super().__init__()
        self._kinematics_solver = ToolFrameKinematics()
        self._state = FlyingGripperState.IDLE

        # control params
        self._control_gain = 0.5
        self._wrist_jog_lookahead_gain = 2.0

        # velocity deadband
        self._velocity_deadband = np.array([
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
        ])
        
        # PID control state
        self._pose_error_integrator = np.zeros(6)
        self._previous_error = np.zeros(6)
        self._first_step = True

        # PID control gains 
        self._kp = np.array([
            1.25,
            1.25,
            1.25,
            1.0,
            1.0,
            1.0,
        ])
        self._kd = np.array([
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ])
        # self._kd = np.zeros(6)
        self._ki = np.array([
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
        ])
        # self._ki = np.zeros(6)

    def reset(self) -> None:
        """
        Resets the PID controller state.
        """
        self._pose_error_integrator = np.zeros(6)
        self._previous_error = np.zeros(6)
        self._first_step = True

    def _compute_error(
        self,
        dt: float,
        current_pos: StretchJointPositions,
        target_pose: pin.SE3,
        target_frame: str = "tool_attachment_site_link"
    ) -> np.ndarray:
        """
        Compute the error between the current pose and the target pose.
        
        Args:
            dt (float): The time step.
            current_pos (StretchJointPositions): The current joint positions.
            target_pose (pin.SE3): The target pose.
            target_frame (str): The name of the target frame.
            
        Returns:
            np.ndarray: The error vector.
        """
        current_pose = self._kinematics_solver.forward(current_pos, target_frame)
        relative_transform = current_pose.actInv(target_pose)
        
        error = pin.log(relative_transform).vector
        
        if self._first_step:
            d_error = np.zeros(6)
            self._first_step = False
        else:
            d_error = (error - self._previous_error) / dt
            
        self._pose_error_integrator += error * dt
        self._previous_error = error

        # Clamp integrator
        self._pose_error_integrator = np.clip(
            self._pose_error_integrator,
            -5.0,
            5.0
        )

        return error, d_error, self._pose_error_integrator

    def _apply_gains(
        self,
        error: np.ndarray,
        d_error: np.ndarray,
        i_error: np.ndarray
    ) -> np.ndarray:
        """
        Apply PID gains to the error.
        
        Args:
            error (np.ndarray): The error vector.
            d_error (np.ndarray): The derivative of the error vector.
            i_error (np.ndarray): The integral of the error vector.
            
        Returns:
            np.ndarray: The velocity vector.
        """
        velocity = np.zeros(6)
        
        # zip up and apply gains
        gains = [self._kp, self._kd, self._ki]
        errors = [error, d_error, i_error]
        
        for i in range(3):
            velocity += gains[i] * errors[i]
        
        # total gain
        velocity *= self._control_gain
        
        return velocity

    def _apply_deadband(
        self,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Apply deadband to the velocity commands.
        
        Args:
            velocity (np.ndarray): The velocity vector.
            
        Returns:
            np.ndarray: The velocity vector.
        """
        velocity = copy.deepcopy(velocity)
        
        for i in range(6):
            if abs(velocity[i]) < self._velocity_deadband[i]:
                velocity[i] = 0.0
        
        return velocity

    def update(
        self,
        dt: float,
        current_pos: StretchJointPositions,
        current_vel: StretchJointVelocities,
        target_pose: pin.SE3
    ) -> tuple[StretchJointVelocities, FlyingGripperState]:
        """
        Core update loop for the flying gripper controller.

        Args:
            dt (float): The time step.
            current_pos (StretchJointPositions): The current joint positions.
            current_vel (StretchJointVelocities): The current joint velocities.
            target_pose (pin.SE3Pose): The target pose as a SE3Pose object.

        Returns:
            A tuple containing:
                - StretchJointVelocities: The enforced velocity commands.
                - FlyingGripperState: The new state.
        """
        error, d_error, i_error = self._compute_error(dt, current_pos, target_pose)
        v_desired = self._apply_gains(error, d_error, i_error)
        v_desired = self._apply_deadband(v_desired)

        dq = self._kinematics_solver.differential_ik(
            current_pos,
            "tool_attachment_site_link",
            v_desired
        )

        # apply lookahead gain to wrist. accounts for move_by lags
        dq.wrist_pitch *= self._wrist_jog_lookahead_gain
        dq.wrist_yaw *= self._wrist_jog_lookahead_gain
        dq.wrist_roll *= self._wrist_jog_lookahead_gain

        dq_limited = self._enforce_joint_velocity_limits(dq)
        
        return dq_limited, self._state
