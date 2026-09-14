import numpy as np
import pinocchio as pin
import pytest

from stretch4_kinematics.state import StretchJointPositions, StretchJointVelocities
from stretch4_kinematics.controllers import FlyingGripperTrackingController, FlyingGripperTrackingState


def test_flying_gripper_tracking_controller_modes():
    controller = FlyingGripperTrackingController()
    dt = 0.05
    current_pos = StretchJointPositions()
    current_vel = StretchJointVelocities()

    # 1. Target SE3 Pose
    target_pose = pin.SE3.Identity()
    target_pose.translation = np.array([0.4, 0.0, 0.6])
    v_cmd, state = controller.update(
        dt=dt,
        current_pos=current_pos,
        current_vel=current_vel,
        target_pose=target_pose
    )
    assert isinstance(v_cmd, StretchJointVelocities)
    assert state == FlyingGripperTrackingState.APPROACH

    controller.reset()

    # 2. Target XYZ Only
    v_cmd, state = controller.update(
        dt=dt,
        current_pos=current_pos,
        current_vel=current_vel,
        target_xyz=np.array([0.4, 0.0, 0.6])
    )
    assert isinstance(v_cmd, StretchJointVelocities)

    controller.reset()

    # 3. Target XYZ + Quat
    v_cmd, state = controller.update(
        dt=dt,
        current_pos=current_pos,
        current_vel=current_vel,
        target_xyz=np.array([0.4, 0.0, 0.6]),
        target_quat=np.array([0.0, 0.0, 0.0, 1.0])
    )
    assert isinstance(v_cmd, StretchJointVelocities)

    controller.reset()

    # 4. Target XYZ + RPY
    v_cmd, state = controller.update(
        dt=dt,
        current_pos=current_pos,
        current_vel=current_vel,
        target_xyz=np.array([0.4, 0.0, 0.6]),
        target_rpy=np.array([0.1, -0.1, 0.2])
    )
    assert isinstance(v_cmd, StretchJointVelocities)


def test_flying_gripper_tracking_controller_missing_target():
    controller = FlyingGripperTrackingController()
    with pytest.raises(ValueError):
        controller.update(
            dt=0.05,
            current_pos=StretchJointPositions(),
            current_vel=StretchJointVelocities()
        )
