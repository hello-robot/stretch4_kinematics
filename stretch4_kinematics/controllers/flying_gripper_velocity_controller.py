import numpy as np

from stretch4_kinematics.state.joint_positions import StretchJointPositions
from stretch4_kinematics.state.joint_velocities import StretchJointVelocities
from stretch4_kinematics.kinematic_models.tool_frame_kinematics import ToolFrameKinematics
from stretch4_kinematics.controllers.base_controllers import StretchVelocityController


class FlyingGripperVelocityController(StretchVelocityController):
    def __init__(self):
        """
        Initializes the FlyingGripperVelocityController, setting up the internal
        kinematics solver.
        """
        super().__init__()
        self._kinematics_solver = ToolFrameKinematics()

    def update(
        self,
        dt: float,
        current_pos: StretchJointPositions,
        current_vel: StretchJointVelocities,
        target_vel: np.ndarray,
        wrist_lookahead_gain: float = 1.0,
    ) -> StretchJointVelocities:
        """
        Computes joint velocities from a desired task space velocity,
        applies wrist lookahead gain, and enforces velocity limits.

        Args:
            dt (float): The time step.
            current_pos (StretchJointPositions): The current joint positions.
            current_vel (StretchJointVelocities): The current joint velocities.
            target_vel (np.ndarray): The desired 6D task space velocity.
            wrist_lookahead_gain (float): Gain factor applied to wrist joint velocities.

        Returns:
            StretchJointVelocities: The limited velocity commands.
        """
        # compute joint velocities
        dq = self._kinematics_solver.differential_ik(
            current_pos,
            "tool_attachment_site_link",
            target_vel
        )

        # apply lookahead gain to wrist. accounts for move_by lags
        dq.wrist_pitch *= wrist_lookahead_gain
        dq.wrist_yaw *= wrist_lookahead_gain
        dq.wrist_roll *= wrist_lookahead_gain

        return self._enforce_joint_velocity_limits(dq)