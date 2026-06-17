import io
import numpy as np
import pinocchio as pin
from yourdfpy import urdf as ud

from stretch4_urdf import get_urdf, get_urdf_calibrated
from stretch4_urdf.utils.urdf_utils_generate_ik_urdfs import (
    add_virtual_rotary_joint,
    add_virtual_planar_joint,
    _make_ik_urdf,
)
from stretch4_kinematics.state.modes import Stretch4IKModes

def _load_stretch4_urdf(
    ik_mode: Stretch4IKModes = Stretch4IKModes.BASE_PLANAR,
    calibrated: bool = False
) -> pin.Model:
    """
    Returns the URDF model for the Stretch 4 robot, modified with a planar base joint,
    clipped joint limits, rigidified non-IK joints, and merged arm joints.
    
    Args:
        ik_mode (Stretch4IKModes): The base mode of the model.
        calibrated (bool): Whether to use the calibrated URDF model.
    
    Returns:
        pin.Model: The Pinocchio model for the Stretch 4 robot.
    """
    if calibrated:
        urdf_string = get_urdf_calibrated(
            model_name="SE4",
            batch_name="francis",
            tool_name="eoa_wrist_dw4_tool_sg4",
        )
    else:
        urdf_string = get_urdf(
            model_name="SE4",
            batch_name="francis",
            tool_name="eoa_wrist_dw4_tool_sg4",
        )

    # Parse URDF string using yourdfpy
    robot_urdf = ud.URDF.load(io.BytesIO(urdf_string.encode('utf-8')))

    # Apply IK modifications: clip joint limits, rigidify non-IK joints, and merge arm
    robot_urdf = _make_ik_urdf(robot_urdf, is_merge_arm=True)

    # Add virtual joint to the base in-place
    if ik_mode == Stretch4IKModes.BASE_ROTATE:
        add_virtual_rotary_joint(robot_urdf)

        joint = robot_urdf.joint_map.get("mobile_base_rotation_joint")
        if joint is not None and joint.limit is not None:
            joint.limit.lower = -1.0 * np.pi
            joint.limit.upper = 1.0 * np.pi
    
    elif ik_mode == Stretch4IKModes.BASE_PLANAR:
        add_virtual_planar_joint(robot_urdf)

    # Write back to XML string
    planar_urdf_string = robot_urdf.write_xml_string()

    # Load the model directly from the XML string
    return pin.buildModelFromXML(planar_urdf_string)


def _compute_limit_interpolation_ratio(
    val: float,
    lower: float,
    upper: float,
    vel: float,
    margin_lower: float,
    margin_upper: float,
) -> float:
    """
    Computes the interpolation ratio (0.0 to 1.0) indicating how far the joint
    has penetrated the limit margin while moving in the direction of that limit.

    If moving in the negative direction and below the lower margin,
    returns a value between 0.0 and 1.0.

    If moving in the positive direction and above the upper margin,
    returns a value between 0.0 and 1.0.

    Otherwise, returns 0.0.

    Args:
        val (float): The current value of the joint position.
        lower (float): The lower limit of the joint position.
        upper (float): The upper limit of the joint position.
        vel (float): The velocity of the joint position.
        margin_lower (float): The margin below the lower limit to start blending.
        margin_upper (float): The margin above the upper limit to start blending.

    Returns:
        float: The interpolation ratio (0.0 to 1.0).
    """
    if vel < 0 and val <= lower + margin_lower:
        return np.clip(((lower + margin_lower) - val) / margin_lower, 0.0, 1.0)
    elif vel > 0 and val >= upper - margin_upper:
        return np.clip((val - (upper - margin_upper)) / margin_upper, 0.0, 1.0)
    return 0.0
