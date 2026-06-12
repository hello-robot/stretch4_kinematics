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
        # Base limits or configurations can go here
        pass

    def _enforce_joint_velocity_limits(self, v_desired: StretchJointVelocities) -> StretchJointVelocities:
        v_limited = copy.deepcopy(v_desired)
        # TODO: Implement limits
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
        self._kp = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

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
        v_limited = self._enforce_joint_velocity_limits(v_desired)

        dq = self._kinematics_solver.differential_ik(
            current_pos,
            "tool_attachment_site_link",
            v_limited
        )
        
        return dq, self._state
