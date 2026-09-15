import io
import numpy as np
import pinocchio as pin
import pytest
import yourdfpy

from stretch4_urdf import get_urdf
from stretch4_kinematics.state import StretchJointPositions, StretchJointVelocities
from stretch4_kinematics.kinematic_models import (
    ToolFrameKinematics,
    PlanarToolFrameKinematics,
    CylindricalToolFrameKinematics,
)


def test_forward_kinematics_consistency():
    urdf_str = get_urdf(
        model_name="SE4",
        batch_name="francis",
        tool_name="eoa_wrist_dw4_tool_sg4",
    )
    urdf_model = yourdfpy.URDF.load(io.StringIO(urdf_str))

    model = pin.buildModelFromXML(urdf_str, pin.JointModelFreeFlyer())
    data = model.createData()
    q = pin.neutral(model)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    tool_frame_id = model.getFrameId("tool_attachment_site_link")
    pin_trans = data.oMf[tool_frame_id].translation

    yourdfpy_transform = urdf_model.get_transform("tool_attachment_site_link", "base_footprint")
    yourdfpy_trans = yourdfpy_transform[:3, 3]

    assert np.allclose(pin_trans, yourdfpy_trans, atol=1e-4)


def test_tool_frame_kinematics_solvers():
    solver = ToolFrameKinematics()
    q_init = StretchJointPositions(arm=0.2, lift=0.5)

    # Test Forward Kinematics
    target_frame = "tool_attachment_site_link"
    pose = solver.forward(q_init, target_frame)
    assert isinstance(pose, pin.SE3)

    # Test Forward Velocity
    q_dot = StretchJointVelocities(arm=0.1, base_theta=0.2)
    v_out = solver.forward_velocity(q_init, q_dot, target_frame)
    assert len(v_out) == 6

    # Test Inverse Kinematics
    target_pose = pin.SE3.Identity()
    target_pose.translation = np.array([0.4, 0.0, 0.6])
    q_sol = solver.inverse_6dof_local(target_frame, target_pose)
    assert isinstance(q_sol, StretchJointPositions)

    # Test Differential IK
    v_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    dq_sol = solver.differential_ik(q_init, target_frame, v_desired)
    assert isinstance(dq_sol, StretchJointVelocities)


def test_not_implemented_subclasses():
    with pytest.raises(NotImplementedError):
        PlanarToolFrameKinematics()

    with pytest.raises(NotImplementedError):
        CylindricalToolFrameKinematics()
