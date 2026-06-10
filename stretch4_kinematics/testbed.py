import io
import numpy as np
import pinocchio as pin
import yourdfpy

from stretch4_urdf import get_urdf, get_urdf_calibrated

from stretch4_kinematics.kinematic_models import (
    ToolFrameKinematics,
    PlanarToolFrameKinematics,
    CylindricalToolFrameKinematics,
)

def test_load_urdf():
    # load yourdfpy with the urdf file from the hello-robot-stretch4-urdf package
    urdf = get_urdf(
        model_name = "SE4",
        batch_name = "francis",
        tool_name = "eoa_wrist_dw4_tool_sg4",
    )

    urdf_model = yourdfpy.URDF.load(io.StringIO(urdf))

    link_names = [link.name for link in urdf_model.robot.links]
    print(f"Found {len(link_names)} links in urdf.")

def test_load_urdf_calibrated():
    urdf = get_urdf_calibrated(
        model_name = "SE4",
        batch_name = "francis",
        tool_name = "eoa_wrist_dw4_tool_sg4",
    )

    urdf_model = yourdfpy.URDF.load(io.StringIO(urdf))
    # urdf_model = yourdfpy.URDF.load(io.BytesIO(urdf.encode('utf-8')))

    link_names = [link.name for link in urdf_model.robot.links]
    print(f"Found {len(link_names)} links in calibrated urdf.")

def test_forward_kinematics():
    urdf = get_urdf(
        model_name = "SE4",
        batch_name = "francis",
        tool_name = "eoa_wrist_dw4_tool_sg4",
    )
    urdf_model = yourdfpy.URDF.load(io.StringIO(urdf))

    # Build Pinocchio model with a FreeFlyer joint to represent the base placement
    model = pin.buildModelFromXML(urdf, pin.JointModelFreeFlyer())
    data = model.createData()

    # Set configuration vector q
    q = pin.neutral(model)

    # Run forward kinematics
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Verify base footprint placement
    base_footprint_id = model.getFrameId("base_footprint")
    base_footprint_placement = data.oMf[base_footprint_id]

    # Compare with yourdfpy for tool_attachment_site_link
    tool_frame_name = "tool_attachment_site_link"
    tool_frame_id = model.getFrameId(tool_frame_name)
    pin_tool_translation = data.oMf[tool_frame_id].translation
    pin_tool_rotation = data.oMf[tool_frame_id].rotation

    # Get relative transform from yourdfpy
    yourdfpy_transform = urdf_model.get_transform(tool_frame_name, "base_footprint")

    print(f"q = {q}")
    print("Translation", pin_tool_translation)
    print("Rotation", pin_tool_rotation)
    print("Yourdfpy Translation", yourdfpy_transform[:3, 3])
    print("Yourdfpy Rotation", yourdfpy_transform[:3, :3])

def test_kinematics_library():
    print("\n--- Testing Kinematics Library ---")
    k1 = ToolFrameKinematics()
    k2 = PlanarToolFrameKinematics()
    k4 = CylindricalToolFrameKinematics()

    print(f"ToolFrameKinematics loaded model: '{k1.model.name}' with {k1.model.nq} joints/coordinates.")
    print(f"PlanarToolFrameKinematics loaded model: '{k2.model.name}' with {k2.model.nq} joints/coordinates.")
    print(f"CylindricalToolFrameKinematics loaded model: '{k4.model.name}' with {k4.model.nq} joints/coordinates.")

    # Define a sample configuration and target frame
    q = pin.neutral(k1.model)
    target_frame = "tool_attachment_site_link"

    print("\nSample Kinematics API Usage:")
    
    # 1. Forward Kinematics Example
    try:
        pose = k1.forward(q, target_frame)
        print(f"  Forward Kinematics: {pose}")
    except NotImplementedError:
        print("  [FK] forward() is defined but not yet implemented.")

    # 2. Inverse Kinematics Example
    try:
        target_pose = pin.SE3.Identity()
        q_sol = k1.inverse(target_frame, target_pose)
        print(f"  Inverse Kinematics: {q_sol}")
    except NotImplementedError:
        print("  [IK] inverse() is defined but not yet implemented.")

    # 3. Differential Kinematics Example
    try:
        # Desired end-effector twist: [v_x, v_y, v_z, w_x, w_y, w_z]
        v_desired = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        dq = k1.differential_ik(q, target_frame, v_desired)
        print(f"  Differential IK: {dq}")
    except NotImplementedError:
        print("  [Diff IK] differential_ik() is defined but not yet implemented.")



###############################################
def main():
    test_load_urdf()
    test_load_urdf_calibrated()
    test_forward_kinematics()
    test_kinematics_library()

if __name__ == '__main__':
    main()