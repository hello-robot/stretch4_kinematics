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


def test_nullspace_projection_joint_space():
    solver = ToolFrameKinematics()
    q_init = StretchJointPositions(arm=0.2, lift=0.5, base_theta=0.1)
    target_frame = "tool_attachment_site_link"

    # Arbitrary joint velocity input
    q_dot_input = StretchJointVelocities(base_x=0.1, base_y=-0.05, arm=0.08, lift=0.02)

    # Compute nullspace projected joint velocities
    v_proj = solver.nullspace_projection_joint_space(q_init, q_dot_input, target_frame=target_frame, damping=0.0)
    assert isinstance(v_proj, StretchJointVelocities)

    # Forward velocity at target frame under projected joint velocity should be near zero
    ee_velocity = solver.forward_velocity(q_init, v_proj, target_frame=target_frame)
    assert np.allclose(ee_velocity, 0.0, atol=1e-3)


def test_hierarchical_differential_ik():
    solver = ToolFrameKinematics()
    q_init = StretchJointPositions(arm=0.25, lift=0.6, base_theta=0.0)
    target_frame = "tool_attachment_site_link"

    v1 = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Secondary task: 2D base velocity [0.1, 0.0]
    J2 = np.zeros((2, solver.model.nv))
    J2[0, 0], J2[0, 1] = 1.0, 0.0
    J2[1, 0], J2[1, 1] = 0.0, 1.0
    v2 = np.array([0.1, 0.0])

    # Case 1: Pass J1 explicitly
    q_pin = q_init.to_pinocchio_q()
    J1_explicit = pin.computeFrameJacobian(
        solver.model,
        solver.data,
        q_pin,
        solver.model.getFrameId(target_frame),
        pin.ReferenceFrame.LOCAL,
    )
    v_cmd_explicit = solver.hierarchical_differential_ik(
        q=q_init, v1=v1, J1=J1_explicit, J2=J2, v2=v2
    )
    assert isinstance(v_cmd_explicit, StretchJointVelocities)
    ee_vel_explicit = solver.forward_velocity(q_init, v_cmd_explicit, target_frame=target_frame)
    assert np.allclose(ee_vel_explicit[:3], v1[:3], atol=1e-2)

    # Case 2: Omit J1 (J1=None), computed automatically via target_frame
    v_cmd_auto = solver.hierarchical_differential_ik(
        q=q_init, v1=v1, target_frame=target_frame, J2=J2, v2=v2
    )
    assert isinstance(v_cmd_auto, StretchJointVelocities)
    assert np.allclose(v_cmd_explicit.to_numpy(), v_cmd_auto.to_numpy(), atol=1e-6)

    # Case 3: Primary task only (J2=None, v2=None)
    v_cmd_primary = solver.hierarchical_differential_ik(
        q=q_init, v1=v1, target_frame=target_frame
    )
    assert isinstance(v_cmd_primary, StretchJointVelocities)


def test_nullspace_base_control():
    solver = ToolFrameKinematics()
    q_init = StretchJointPositions(arm=0.25, lift=0.6, base_theta=0.0)
    target_frame = "tool_attachment_site_link"

    # Drive base translation in world XY while keeping EE fixed in space
    v_base_world_des = np.array([0.1, 0.05])
    v_cmd = solver.nullspace_base_control(
        q=q_init,
        v_ee_local_desired=np.zeros(6),
        v_base_world_desired=v_base_world_des,
        target_frame=target_frame,
    )
    assert isinstance(v_cmd, StretchJointVelocities)

    # Check that EE spatial velocity remains close to zero
    ee_velocity = solver.forward_velocity(q_init, v_cmd, target_frame=target_frame)
    assert np.allclose(ee_velocity, 0.0, atol=1e-2)


def test_not_implemented_subclasses():
    with pytest.raises(NotImplementedError):
        PlanarToolFrameKinematics()

    with pytest.raises(NotImplementedError):
        CylindricalToolFrameKinematics()
