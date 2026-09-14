import numpy as np
import pinocchio as pin

from stretch4_kinematics.state import (
    StretchJointPositions,
    StretchJointVelocities,
    Stretch4IKModes,
)


def test_joint_positions_conversions():
    q_pos = StretchJointPositions(
        base_x=0.1,
        base_y=-0.2,
        base_theta=0.5,
        lift=0.6,
        arm=0.25,
        wrist_yaw=0.1,
        wrist_pitch=-0.3,
        wrist_roll=0.4,
    )

    # Test to_numpy / from_numpy
    arr = q_pos.to_numpy()
    assert len(arr) == 8
    q_reconstructed = StretchJointPositions.from_numpy(arr)
    assert np.isclose(q_reconstructed.base_x, 0.1)
    assert np.isclose(q_reconstructed.lift, 0.6)
    assert np.isclose(q_reconstructed.arm, 0.25)

    # Test Pinocchio q conversion
    pin_q = q_pos.to_pinocchio_q()
    assert isinstance(pin_q, np.ndarray)
    q_from_pin = StretchJointPositions.from_pinocchio_q(pin_q)
    assert np.isclose(q_from_pin.base_x, 0.1)
    assert np.isclose(q_from_pin.wrist_roll, 0.4)

    # Test joint names
    names = q_pos.get_joint_names()
    assert len(names) == 8
    assert names[0] == "base_x"
    assert names[-1] == "wrist_roll"


def test_joint_velocities_conversions():
    v_vel = StretchJointVelocities(
        base_x=0.05,
        base_y=-0.05,
        base_theta=0.1,
        lift=0.02,
        arm=0.01,
        wrist_yaw=0.05,
        wrist_pitch=-0.05,
        wrist_roll=0.05,
    )

    arr = v_vel.to_numpy()
    assert len(arr) == 8
    v_reconstructed = StretchJointVelocities.from_numpy(arr)
    assert np.isclose(v_reconstructed.base_x, 0.05)
    assert np.isclose(v_reconstructed.arm, 0.01)


def test_ik_modes_enum():
    assert Stretch4IKModes.BASE_FIXED.value == 1
    assert Stretch4IKModes.BASE_ROTATE.value == 2
    assert Stretch4IKModes.BASE_PLANAR.value == 3
