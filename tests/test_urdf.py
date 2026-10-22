import io
import yourdfpy
from stretch4_urdf import get_urdf, get_urdf_calibrated


def test_load_urdf():
    urdf_str = get_urdf(
        model_name="SE4",
        batch_name="francis",
        tool_name="eoa_wrist_dw4_tool_sg4",
    )
    urdf_model = yourdfpy.URDF.load(io.StringIO(urdf_str))
    link_names = [link.name for link in urdf_model.robot.links]
    assert len(link_names) > 0
    assert "base_footprint" in link_names
    assert "tool_attachment_site_link" in link_names


def test_load_urdf_calibrated():
    urdf_str = get_urdf_calibrated(
        model_name="SE4",
        batch_name="francis",
        tool_name="eoa_wrist_dw4_tool_sg4",
    )
    urdf_model = yourdfpy.URDF.load(io.StringIO(urdf_str))
    link_names = [link.name for link in urdf_model.robot.links]
    assert len(link_names) > 0
    assert "base_footprint" in link_names


import numpy as np
import pytest

from stretch4_kinematics import StretchJointPositions, StretchKinematics


@pytest.mark.parametrize("tool_name", ["eoa_wrist_dw4_tool_sg4", "eoa_wrist_dw4_tool_pg4"])
def test_merged_arm_matches_the_urdf(tool_name):
    """
    The IK model's single arm joint puts the tool where the URDF's four telescoping joints do.
    (stretch4_urdf's merge_arm() used to leave it ~41 mm further out.)
    """
    urdf = yourdfpy.URDF.load(io.StringIO(get_urdf(model_name="SE4", batch_name="francis", tool_name=tool_name)))
    kinematics = StretchKinematics(tool_name=tool_name)
    rng = np.random.default_rng(0)
    for _ in range(10):
        lift, arm = rng.uniform(0.1, 1.1), rng.uniform(0.0, 0.5)
        yaw, pitch, roll = rng.uniform(-1.0, 3.0), rng.uniform(-1.0, 1.5), rng.uniform(-3.0, 1.0)
        cfg = {"lift_joint": lift, "wrist_yaw_joint": yaw, "wrist_pitch_joint": pitch, "wrist_roll_joint": roll}
        cfg.update({f"arm_l{i}_joint": arm / 4 for i in range(1, 5)})
        urdf.update_cfg(cfg)
        expected = urdf.get_transform("grasp_center_link", "base_footprint")

        q = StretchJointPositions(0.0, 0.0, 0.0, lift, arm, yaw, pitch, roll)
        actual = kinematics.forward(q, "grasp_center_link").homogeneous
        np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=1e-5)
        np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=1e-5)
