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

ARM_JOINTS = ("arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint")

WRIST_ROLL_LIMITS = (np.deg2rad(-65.0), np.deg2rad(245.0))
"""The wrist roll's travel on the robot (stretch4_body `SE4_wrist_roll_DW4` range_deg) and in
stretch4_mujoco (`FLIP_WRIST_ROLL_RANGE`). The URDF has it mirrored, as [-245, 65] degrees."""

DEFAULT_TOOL_NAME = "eoa_wrist_dw4_tool_sg4"
SUPPORTED_TOOL_NAMES = ("eoa_wrist_dw4_tool_sg4", "eoa_wrist_dw4_tool_pg4")

def _load_stretch4_urdf(
    ik_mode: Stretch4IKModes = Stretch4IKModes.BASE_PLANAR,
    calibrated: bool = False,
    tool_name: str = DEFAULT_TOOL_NAME,
) -> pin.Model:
    """
    Returns the URDF model for the Stretch 4 robot, modified with a planar base joint,
    clipped joint limits, rigidified non-IK joints, and merged arm joints.
    
    Args:
        ik_mode (Stretch4IKModes): The base mode of the model.
        calibrated (bool): Whether to use the calibrated URDF model.
        tool_name (str): The end of arm tool, one of SUPPORTED_TOOL_NAMES.
    
    Returns:
        pin.Model: The Pinocchio model for the Stretch 4 robot.
    """
    if tool_name not in SUPPORTED_TOOL_NAMES:
        raise ValueError(
            f"Unsupported tool_name {tool_name!r}; expected one of {SUPPORTED_TOOL_NAMES}."
        )

    if calibrated:
        urdf_string = get_urdf_calibrated(
            model_name="SE4",
            batch_name="francis",
            tool_name=tool_name,
        )
    else:
        urdf_string = get_urdf(
            model_name="SE4",
            batch_name="francis",
            tool_name=tool_name,
        )

    # Parse URDF string using yourdfpy
    robot_urdf = ud.URDF.load(io.BytesIO(urdf_string.encode('utf-8')))

    # Apply IK modifications: clip joint limits and rigidify non-IK joints. The arm is merged
    # here rather than by stretch4_urdf's merge_arm(), which drops the inner links' offsets.
    robot_urdf = _make_ik_urdf(robot_urdf, is_merge_arm=False)
    _merge_arm(robot_urdf)
    roll = robot_urdf.joint_map["wrist_roll_joint"]
    roll.limit.lower, roll.limit.upper = WRIST_ROLL_LIMITS

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


def _merge_arm(robot_urdf: ud.URDF) -> None:
    """
    Replaces the four telescoping arm joints with one prismatic joint, in place, so that IK
    sees the arm as a single extension.

    The last arm joint is reattached to the first one's parent with all four joints' origins
    composed into its own, and the travel of all four. The middle three become fixed.

    stretch4_urdf's `merge_arm()` does the same but sums `joint.origin[3, :3]`, the bottom row
    of the homogeneous transform (always zero), so the merged joint keeps only the last link's
    offset and the tool ends up ~41 mm further out than the real arm puts it.
    """
    joints = [robot_urdf.joint_map[name] for name in ARM_JOINTS]
    first, last = joints[0], joints[-1]

    origin = np.eye(4)
    travel = 0.0
    for joint in joints:
        origin = origin @ (joint.origin if joint.origin is not None else np.eye(4))
        if joint.limit is not None:
            travel += joint.limit.upper

    last.parent = first.parent
    last.origin = origin
    last.limit.upper = travel
    for joint in joints[:-1]:
        joint.type = "fixed"


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
