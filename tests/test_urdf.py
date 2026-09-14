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
