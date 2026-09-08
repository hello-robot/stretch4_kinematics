#!/usr/bin/env python3
import sys
import time
import numpy as np

from stretch4_body.robot.robot_client import RobotClient
from stretch4_kinematics.stretch_interface import StretchInterface
import stretch4_kinematics.kinematic_models.tool_frame_kinematics as toolframekinematics
from stretch4_kinematics.kinematic_models.tool_frame_kinematics import ToolFrameKinematics
from stretch4_kinematics.state import (
    StretchJointPositions,
    StretchJointVelocities,
)


def main():
    print("=== Stretch 4 Nullspace Motion Demo ===")
    print("Moving base in a circle (X and Y components) with radius 0.2m and period 10s")
    print("Velocities start at (0, 0) and target vector is passed into nullspace_projection.")
    print("Press Ctrl+C to stop.\n")

    # 1. Instantiate robot client & stretch interface
    robot = RobotClient()
    robot.startup()
    interface = StretchInterface(robot=robot)
    interface.reset_odometry_offset()

    # 2. Instantiate kinematics model
    kinematics = ToolFrameKinematics()

    # Motion parameters
    radius = 0.3     # meters
    period = 10.0    # seconds
    omega = 2.0 * np.pi / period  # rad/s

    gain_y = 5.

    rate_hz = 20.0
    dt = 1.0 / rate_hz

    start_time = time.time()

    try:
        while True:
            loop_start = time.time()
            elapsed = loop_start - start_time

            # Parameterize circle angle theta(t) with zero initial angular velocity:
            # theta(t) = omega * t - sin(omega * t)
            # dtheta/dt = omega * (1 - cos(omega * t))
            theta = omega * elapsed - np.sin(omega * elapsed)
            dtheta_dt = omega * (1.0 - np.cos(omega * elapsed))

            # Circle position parameterization: x(theta) = R*(1 - cos(theta)), y(theta) = R*sin(theta)
            # Differentiating gives velocities starting at (0, 0) at t = 0:
            v_x = 0
            v_y = gain_y * radius * np.cos(theta) * dtheta_dt

            # Target joint velocity vector (X and Y velocities)
            target_q_dot = StretchJointVelocities(base_x=v_x, base_y=v_y)

            # Query current joint positions
            current_pos = interface.get_joint_position()

            # Compute nullspace projection of the target velocity
            v_proj_np = kinematics.nullspace_projection(current_pos, target_q_dot)
            v_cmd = StretchJointVelocities.from_numpy(v_proj_np)

            # Command projected velocities to the robot rounded to 3 decimal places
            print(f"v_desired: {np.round(target_q_dot.to_numpy(), 3)}")
            print(f"v_proj_np: {np.round(v_proj_np, 3)}")
            interface.cmd_velocities(v_cmd)

            # Sleep to maintain control loop frequency
            execution_time = time.time() - loop_start
            sleep_time = max(0.0, dt - execution_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting cleanly...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Stopping robot and shutting down interface...")
        try:
            interface.cmd_zero_velocity()
            robot.stop()
        except Exception as shutdown_err:
            print(f"Error during robot shutdown: {shutdown_err}")
        print("Shutdown complete.")


if __name__ == '__main__':
    main()
