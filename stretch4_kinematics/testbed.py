import io
import time

import numpy as np
import pinocchio as pin
import yourdfpy

from stretch4_urdf import get_urdf, get_urdf_calibrated

from stretch4_kinematics.state import (
    StretchJointPositions,
    StretchJointVelocities,
)
from stretch4_kinematics.kinematic_models import (
    ToolFrameKinematics,
    PlanarToolFrameKinematics,
    CylindricalToolFrameKinematics,
)
from stretch4_kinematics.controllers.flying_gripper_tracking_controller import FlyingGripperTrackingController


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
        joint_position = StretchJointPositions.from_pinocchio_q(q)
        pose = k1.forward(joint_position, target_frame)
        print(f"  Forward Kinematics: {pose}")
    except NotImplementedError:
        print("  [FK] forward() is defined but not yet implemented.")

    # 2. Inverse Kinematics Example
    try:
        target_pose = pin.SE3.Identity()
        target_pose.translation = np.array([0.3, 0.1, 0.5])
        target_pose.rotation = np.eye(3)
        q_sol = k1.inverse_6dof_local(target_frame, target_pose)
        print(f"  Inverse Kinematics to {target_frame} at pose {target_pose} (SE3):")
        q_sol.pretty_print()
    except NotImplementedError:
        print("  [IK] inverse() is defined but not yet implemented.")

    # 3. Differential Kinematics Example
    try:
        # Desired end-effector linear velocity in tool frame: [v_forward, v_left, v_up]
        v_desired = np.array([0.1, 0.05, -0.1, 0.0, 0.0, 0.0])
        dq = k1.differential_ik(joint_position, target_frame, v_desired)
        print("  Differential IK:")
        dq.pretty_print()
    except NotImplementedError:
        print("  [Diff IK] differential_ik() is defined but not yet implemented.")

def test_controls_library():
    print("\n--- Testing Controls Library ---")
    controller = FlyingGripperTrackingController()
    print("FlyingGripperTrackingController successfully initialized.")

    dt = 0.05
    # Initialize joint state
    current_pos = StretchJointPositions()
    current_vel = StretchJointVelocities()

    # 1. Test target_pose input (pin.SE3)
    print("\nTesting input: target_pose (pin.SE3)")
    target_pose = pin.SE3.Identity()
    target_pose.translation = np.array([0.3, 0.1, 0.5])
    target_pose.rotation = np.eye(3)
    
    v_cmd, state = controller.update(
        dt=dt,
        current_pos=current_pos,
        current_vel=current_vel,
        target_pose=target_pose
    )
    print(f"  Controller state: {state}")
    print("  Commanded velocities:")
    v_cmd.pretty_print()

    # Reset controller state for next test
    controller.reset()

    # 2. Test target_xyz input only
    print("\nTesting input: target_xyz only")
    target_xyz = np.array([0.3, 0.1, 0.5])
    v_cmd, state = controller.update(
        dt=dt,
        current_pos=current_pos,
        current_vel=current_vel,
        target_xyz=target_xyz
    )
    print(f"  Controller state: {state}")
    print("  Commanded velocities:")
    v_cmd.pretty_print()

    controller.reset()

    # 3. Test target_xyz + target_quat input
    print("\nTesting input: target_xyz + target_quat")
    target_quat = np.array([0.0, 0.0, 0.0, 1.0])
    v_cmd, state = controller.update(
        dt=dt,
        current_pos=current_pos,
        current_vel=current_vel,
        target_xyz=target_xyz,
        target_quat=target_quat
    )
    print(f"  Controller state: {state}")
    print("  Commanded velocities:")
    v_cmd.pretty_print()

    controller.reset()

    # 4. Test target_xyz + target_rpy input
    print("\nTesting input: target_xyz + target_rpy")
    target_rpy = np.array([0.1, -0.2, 0.3])
    v_cmd, state = controller.update(
        dt=dt,
        current_pos=current_pos,
        current_vel=current_vel,
        target_xyz=target_xyz,
        target_rpy=target_rpy
    )
    print(f"  Controller state: {state}")
    print("  Commanded velocities:")
    v_cmd.pretty_print()

    controller.reset()

    # 5. Test invalid input (both target_pose and target_xyz are None)
    print("\nTesting input: invalid (no target)")
    try:
        controller.update(
            dt=dt,
            current_pos=current_pos,
            current_vel=current_vel
        )
        print("  WARNING: Expected ValueError, but no exception was raised!")
    except ValueError as e:
        print(f"  Success: Raised expected ValueError: {e}")

def test_nullspace_projection():
    q = StretchJointPositions()
    kinematics = ToolFrameKinematics()
    kinematics.diff_ik_testbed(q)


def test_nullspace_base_square(
    numerical: bool = True,
    side_length: float = 0.20,
    speed: float = 0.10,
    rate_hz: float = 50.0,
    kp_ee: float = 2.0,
) -> None:
    """
    Drive the mobile base in a square trajectory in the world XY plane from its
    current position while keeping the end-effector stationary using nullspace control.

    Args:
        numerical: If True, integrates velocities in numerical simulation without
            moving the physical robot. If False, connects to RobotClient and commands
            hardware velocities.
        side_length: Length of each side of the square in meters.
        speed: Desired linear speed of the base in meters per second.
        rate_hz: Control loop frequency in Hertz.
        kp_ee: Proportional feedback gain on the local end-effector pose error to
            prevent integration or hardware drift.

    Returns:
        None.
    """
    kinematics = ToolFrameKinematics()
    target_frame = "tool_attachment_site_link"
    dt = 1.0 / rate_hz
    side_duration = side_length / speed

    robot = None
    interface = None
    if numerical:
        # Start with the arm partially extended so it has stroke margin in all directions
        current_pos = StretchJointPositions(
            base_x=0.0,
            base_y=0.0,
            base_theta=0.0,
            lift=0.5,
            arm=0.25,
            wrist_yaw=0.0,
            wrist_pitch=0.0,
            wrist_roll=0.0,
        )
    else:
        from stretch4_body.robot.robot_client import RobotClient
        from stretch4_kinematics.stretch_interface import StretchInterface

        robot = RobotClient()
        if not robot.startup():
            raise RuntimeError("Failed to start RobotClient connection.")
        interface = StretchInterface(robot=robot)
        interface.reset_odometry_offset()
        current_pos = interface.get_joint_position()

    initial_base_xy = np.array([current_pos.base_x, current_pos.base_y])
    initial_ee_pose = kinematics.forward(current_pos, target_frame)

    # Four cardinal directions in the world XY plane forming a closed 20cm x 20cm square
    square_directions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]

    max_ee_pos_err = 0.0
    max_ee_rot_err = 0.0

    print(
        f"\n--- Running Nullspace Base Square ({side_length:.2f}m x {side_length:.2f}m "
        f"at {speed:.2f}m/s, numerical={numerical}) ---"
    )
    print(f"Initial Base XY: {np.round(initial_base_xy, 4)}")
    print(f"Initial EE XYZ:  {np.round(initial_ee_pose.translation, 4)}")

    try:
        n = 1 if numerical else 5
        for _ in range(n):
            for side_idx, direction in enumerate(square_directions):
                v_base_world_des = direction * speed
                steps_per_side = int(round(side_duration / dt))

                for _ in range(steps_per_side):
                    loop_start = time.time()

                    if not numerical and interface is not None:
                        current_pos = interface.get_joint_position()

                    # Compute small closed-loop corrective twist in the local EE frame
                    # to eliminate numerical integration or hardware tracking drift
                    current_ee_pose = kinematics.forward(current_pos, target_frame)
                    ee_err_local = pin.log(current_ee_pose.actInv(initial_ee_pose)).vector
                    v_ee_local_des = kp_ee * ee_err_local

                    pos_err = float(np.linalg.norm(current_ee_pose.translation - initial_ee_pose.translation))
                    rot_err = float(np.linalg.norm(ee_err_local[3:]))
                    max_ee_pos_err = max(max_ee_pos_err, pos_err)
                    max_ee_rot_err = max(max_ee_rot_err, rot_err)

                    v_cmd = kinematics.nullspace_base_control(
                        q=current_pos,
                        v_ee_local_des=v_ee_local_des,
                        v_base_world_des=v_base_world_des,
                    )

                    if numerical:
                        theta = current_pos.base_theta
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)
                        world_vx = v_cmd.base_x * cos_t - v_cmd.base_y * sin_t
                        world_vy = v_cmd.base_x * sin_t + v_cmd.base_y * cos_t

                        q_next = current_pos.to_numpy()
                        q_next[0] += world_vx * dt
                        q_next[1] += world_vy * dt
                        q_next[2] += v_cmd.base_theta * dt
                        q_next[3:] += v_cmd.to_numpy()[3:] * dt
                        current_pos = StretchJointPositions.from_numpy(q_next)
                    elif interface is not None:
                        interface.cmd_velocities(v_cmd)
                        elapsed = time.time() - loop_start
                        if dt > elapsed:
                            time.sleep(dt - elapsed)

                corner_ee_pose = kinematics.forward(current_pos, target_frame)
                print(
                    f"Corner {side_idx + 1}: Base XY = [{current_pos.base_x:.4f}, {current_pos.base_y:.4f}], "
                    f"Base Theta = {current_pos.base_theta:.4f} rad, Arm = {current_pos.arm:.4f} m | "
                    f"EE XYZ = {np.round(corner_ee_pose.translation, 4)}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if not numerical and interface is not None and robot is not None:
            interface.cmd_zero_velocity()
            robot.stop()

    final_ee_pose = kinematics.forward(current_pos, target_frame)
    print(f"Final Base XY:   [{current_pos.base_x:.4f}, {current_pos.base_y:.4f}]")
    print(f"Final EE XYZ:    {np.round(final_ee_pose.translation, 4)}")
    print(f"Max EE Pos Drift: {max_ee_pos_err:.6f} m, Max EE Rot Drift: {max_ee_rot_err:.6f} rad")


###############################################
def main():
    # test_load_urdf()
    # test_load_urdf_calibrated()
    # test_forward_kinematics()
    # test_kinematics_library()
    # test_controls_library()
    # test_nullspace_projection()
    test_nullspace_base_square(numerical=False)

if __name__ == '__main__':
    main()