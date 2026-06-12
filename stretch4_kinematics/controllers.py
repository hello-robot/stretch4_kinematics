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
        self._kp = np.array([
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.01,
        ])

    def _compute_error(
        self,
        current_pos: StretchJointPositions,
        target_pose: pin.SE3,
        target_frame: str = "tool_attachment_site_link"
    ) -> np.ndarray:
        # 1. Get current pose using forward kinematics
        current_pose = self._kinematics_solver.forward(current_pos, target_frame)
        
        # 2. Compute the relative transform from current to target: T_c^-1 * T_t
        relative_transform = current_pose.actInv(target_pose)
        
        # 3. Compute 6D twist error [v, w] using the Lie algebra log map
        # err[:3] is the translation error vector in the gripper's local frame
        # err[3:] is the rotation error vector in the gripper's local frame
        err = pin.log(relative_transform).vector
        
        return err
        
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
        error = self._compute_error(current_pos, target_pose)
        v_desired = self._kp * error

        dq = self._kinematics_solver.differential_ik(
            current_pos,
            "tool_attachment_site_link",
            v_desired
        )

        dq_limited = self._enforce_joint_velocity_limits(dq)
        
        return dq_limited, self._state
