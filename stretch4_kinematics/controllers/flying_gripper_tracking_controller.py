import copy
from enum import Enum, auto
import numpy as np
import pinocchio as pin

from stretch4_kinematics.state.joint_positions import StretchJointPositions
from stretch4_kinematics.state.joint_velocities import StretchJointVelocities
from stretch4_kinematics.controllers.base_controllers import StretchTrackingController
from stretch4_kinematics.controllers.flying_gripper_velocity_controller import FlyingGripperVelocityController

class FlyingGripperTrackingState(Enum):
    """
    Flying gripper controller states.
    """
    IDLE = auto()
    APPROACH = auto()
    FINISH = auto()
    FAIL = auto()


class FlyingGripperTrackingController(StretchTrackingController):
    def __init__(self):
        """
        Initializes the FlyingGripperController, setting up the PID gains,
        integration states, deadband thresholds, and the internal velocity controller.
        """
        super().__init__()
        self._velocity_controller = FlyingGripperVelocityController()
        # Keep references for backward compatibility / internal calculations
        self._kinematics_solver = self._velocity_controller._kinematics_solver
        self._state = FlyingGripperTrackingState.IDLE

        # task params
        self._translation_tolerance = 0.01  # meters
        # self._rotation_tolerance = np.deg2rad(2.0)  # rad  # UNUSED

        # control params
        self._control_gain = 0.5
        self._wrist_jog_lookahead_gain = 1.0

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
        self._ki = np.array([
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
        ])

    def reset(self) -> None:
        """
        Resets the PID controller state.
        """
        self._pose_error_integrator = np.zeros(6)
        self._previous_error = np.zeros(6)
        self._first_step = True
        self._state = FlyingGripperTrackingState.IDLE

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
        
        # Decouple translation error from orientation warping
        error = np.zeros(6)
        error[:3] = relative_transform.translation
        error[3:] = pin.log(relative_transform).vector[3:]
        
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
        target_pose: pin.SE3 = None,
        target_xyz: np.ndarray = None,
        target_quat: np.ndarray = None,
        target_rpy: np.ndarray = None,
    ) -> tuple[StretchJointVelocities, FlyingGripperTrackingState]:
        """
        Core update loop for the flying gripper controller.

        Args:
            dt (float): The time step.
            current_pos (StretchJointPositions): The current joint positions.
            current_vel (StretchJointVelocities): The current joint velocities.
            target_pose (pin.SE3, optional): The target pose as a SE3 object.
            target_xyz (np.ndarray, optional): The desired position of the target frame in the world frame.
            target_quat (np.ndarray, optional): The desired orientation of the target frame as a quaternion (scalar-last: [x, y, z, w]).
            target_rpy (np.ndarray, optional): The desired orientation of the target frame as RPY angles (radians).

        Returns:
            A tuple containing:
                - StretchJointVelocities: The enforced velocity commands.
                - FlyingGripperState: The new state.
        """
        # get target pose
        target_pose = self._parse_target_input(
            target_pose=target_pose,
            target_xyz=target_xyz,
            target_quat=target_quat,
            target_rpy=target_rpy
        )

        # compute error and check termination
        error, d_error, i_error = self._compute_error(dt, current_pos, target_pose)
        translation_err = np.linalg.norm(error[:3])

        # check if finished
        if translation_err < self._translation_tolerance:
            self._state = FlyingGripperTrackingState.FINISH
            return StretchJointVelocities(), self._state
        else:
            self._state = FlyingGripperTrackingState.APPROACH

        # compute desired velocity
        v_desired = self._apply_gains(error, d_error, i_error)
        v_desired = self._apply_deadband(v_desired)

        # delegate to direct velocity controller
        dq_limited = self._velocity_controller.update(
            dt,
            current_pos,
            current_vel,
            v_desired,
            wrist_lookahead_gain=self._wrist_jog_lookahead_gain
        )

        return dq_limited, self._state
