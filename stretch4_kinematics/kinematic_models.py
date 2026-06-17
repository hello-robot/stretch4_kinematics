from dataclasses import dataclass, fields
from enum import Enum, auto

import io
import numpy as np
import pinocchio as pin
from yourdfpy import urdf as ud

from stretch4_urdf import get_urdf, get_urdf_calibrated
from stretch4_urdf.utils.urdf_utils_generate_ik_urdfs import (
    add_virtual_rotary_joint,
    add_virtual_planar_joint,
    _make_ik_urdf,  # TODO: change name to be public
)

class Stretch4IKModes(Enum):
    """
    Represents the different URDF configurations for the Stretch 4 URDF.
    """
    BASE_FIXED = auto()   # base cannot move
    BASE_ROTATE = auto()  # base can only rotate about z-axis
    BASE_PLANAR = auto()  # base can translate and rotate (SE(2))


def _load_stretch4_urdf(
    ik_mode: Stretch4IKModes = Stretch4IKModes.BASE_PLANAR,
    calibrated: bool = False
) -> pin.Model:
    """
    Returns the URDF model for the Stretch 4 robot, modified with a planar base joint,
    clipped joint limits, rigidified non-IK joints, and merged arm joints.
    
    Args:
        calibrated (bool): Whether to use the calibrated URDF model.
    
    Returns:
        pin.Model: The Pinocchio model for the Stretch 4 robot.
    """
    if calibrated:
        urdf_string = get_urdf_calibrated(
            model_name = "SE4",
            batch_name = "francis",
            tool_name = "eoa_wrist_dw4_tool_sg4",
        )
    else:
        urdf_string = get_urdf(
            model_name = "SE4",
            batch_name = "francis",
            tool_name = "eoa_wrist_dw4_tool_sg4",
        )

    # Parse URDF string using yourdfpy
    robot_urdf = ud.URDF.load(io.BytesIO(urdf_string.encode('utf-8')))

    # Apply IK modifications: clip joint limits, rigidify non-IK joints, and merge arm
    # TODO don't use this internal func
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



@dataclass
class StretchJointPositions:
    """
    Represents the 8-element joint positions of Stretch 4.
    Models the omnidirectional base as SE(2): two translations and a rotation.

    Contains helper functions to convert to/from:
        - Pinocchio's 9-element configuration vector q
        - 8-element numpy array
        - 8-element dictionary
    """
    # Mobile Base (physical translations and rotation)
    base_x: float = 0.0
    base_y: float = 0.0
    base_theta: float = 0.0  # Physical angle in radians
    
    # Arm translation
    lift: float = 0.5
    arm: float = 0.0  # Merged arm extension (meters)
    
    # End of Arm (wrist joints in radians)
    wrist_yaw: float = 0.0
    wrist_pitch: float = 0.0
    wrist_roll: float = 0.0

    def get_joint_names(self) -> list[str]:
        """
        Returns the joint names in the order corresponding to the numpy array.
        
        Returns:
            list[str]: The joint names.
        """
        return [
            "base_x",
            "base_y",
            "base_theta",
            "lift",
            "arm",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        ]

    def to_pinocchio_q(
        self,
        mode: Stretch4IKModes = Stretch4IKModes.BASE_PLANAR
    ) -> np.ndarray:
        """
        Converts the physical joint state into Pinocchio's configuration vector `q`
        based on the selected IK base mode.

        Args:
            mode (Stretch4IKModes): The base mode of the model.

        Returns:
            np.ndarray: Pinocchio's configuration vector.
        """
        if mode == Stretch4IKModes.BASE_PLANAR:
            return np.array([
                self.base_x,
                self.base_y,
                np.cos(self.base_theta),
                np.sin(self.base_theta),
                self.lift,
                self.arm,
                self.wrist_yaw,
                self.wrist_pitch,
                self.wrist_roll
            ])
        elif mode == Stretch4IKModes.BASE_ROTATE:
            return np.array([
                self.base_theta,
                self.lift,
                self.arm,
                self.wrist_yaw,
                self.wrist_pitch,
                self.wrist_roll
            ])
        elif mode == Stretch4IKModes.BASE_FIXED:
            return np.array([
                self.lift,
                self.arm,
                self.wrist_yaw,
                self.wrist_pitch,
                self.wrist_roll
            ])
        else:
            raise ValueError(f"Unsupported IK mode: {mode}")

    @classmethod
    def from_pinocchio_q(
        cls,
        q: np.ndarray,
        mode: Stretch4IKModes = None
    ) -> "StretchJointPositions":
        """
        Creates a StretchJointPositions instance from a Pinocchio configuration vector `q`.
        Auto-detects the mode if not specified, based on the length of q.

        Args:
            q (np.ndarray): Pinocchio's configuration vector.
            mode (Stretch4IKModes, optional): The base mode of the model (handles variable length)

        Returns:
            StretchJointPositions: Instance of StretchJointPositions.
        """
        if mode is None:
            if len(q) == 9:
                mode = Stretch4IKModes.BASE_PLANAR
            elif len(q) == 6:
                mode = Stretch4IKModes.BASE_ROTATE
            elif len(q) == 5:
                mode = Stretch4IKModes.BASE_FIXED
            else:
                raise ValueError(f"Cannot auto-detect IK mode for configuration vector of length {len(q)}")

        if mode == Stretch4IKModes.BASE_PLANAR:
            # Recover physical theta from cos(theta) (q[2]) and sin(theta) (q[3])
            theta = np.arctan2(q[3], q[2])
            return cls(
                base_x=q[0],
                base_y=q[1],
                base_theta=theta,
                lift=q[4],
                arm=q[5],
                wrist_yaw=q[6],
                wrist_pitch=q[7],
                wrist_roll=q[8]
            )
        elif mode == Stretch4IKModes.BASE_ROTATE:
            return cls(
                base_x=0.0,
                base_y=0.0,
                base_theta=q[0],
                lift=q[1],
                arm=q[2],
                wrist_yaw=q[3],
                wrist_pitch=q[4],
                wrist_roll=q[5]
            )
        elif mode == Stretch4IKModes.BASE_FIXED:
            return cls(
                base_x=0.0,
                base_y=0.0,
                base_theta=0.0,
                lift=q[0],
                arm=q[1],
                wrist_yaw=q[2],
                wrist_pitch=q[3],
                wrist_roll=q[4]
            )
        else:
            raise ValueError(f"Unsupported IK mode: {mode}")

    def to_numpy(self) -> np.ndarray:
        """
        Converts the joint positions to a standard 8-element NumPy array.

        Returns:
            np.ndarray: The 8-element joint configuration vector.
        """
        return np.array([
            self.base_x,
            self.base_y,
            self.base_theta,
            self.lift,
            self.arm,
            self.wrist_yaw,
            self.wrist_pitch,
            self.wrist_roll,
        ])

    @classmethod
    def from_numpy(cls, q: np.ndarray) -> "StretchJointPositions":
        """
        Creates a StretchJointPositions instance from an 8-element joint configuration vector `q`.

        Args:
            q (np.ndarray): The 8-element joint configuration vector.

        Returns:
            StretchJointPositions: Instance of StretchJointPositions.
        """
        return cls(
            base_x=q[0],
            base_y=q[1],
            base_theta=q[2],
            lift=q[3],
            arm=q[4],
            wrist_yaw=q[5],
            wrist_pitch=q[6],
            wrist_roll=q[7],
        )

    def to_dict(self) -> dict:
        """
        Converts the joint positions to an 8-element configuration dictionary.

        Returns:
            dict: The 8-element joint configuration dictionary.
        """
        return {
            "base_x": self.base_x,
            "base_y": self.base_y,
            "base_theta": self.base_theta,
            "lift": self.lift,
            "arm": self.arm,
            "wrist_yaw": self.wrist_yaw,
            "wrist_pitch": self.wrist_pitch,
            "wrist_roll": self.wrist_roll,
        }

    @classmethod
    def from_dict(cls, joint_dict: dict) -> "StretchJointPositions":
        """
        Creates a StretchJointPositions instance from a dictionary of joint configurations.

        Args:
            joint_dict (dict): Dictionary containing the joint configurations.

        Returns:
            StretchJointPositions: Instance of StretchJointPositions.
        """
        return cls(
            base_x=joint_dict["base_x"],
            base_y=joint_dict["base_y"],
            base_theta=joint_dict["base_theta"],
            lift=joint_dict["lift"],
            arm=joint_dict["arm"],
            wrist_yaw=joint_dict["wrist_yaw"],
            wrist_pitch=joint_dict["wrist_pitch"],
            wrist_roll=joint_dict["wrist_roll"],
        )

    def pretty_print(self) -> None:
        """
        Pretty-prints each joint name and value on sequential lines.
        """
        for field in fields(self):
            print(f"{field.name}: {getattr(self, field.name):.4f}")


@dataclass
class StretchJointVelocities:
    """
    Represents the 8-element joint velocity vector of the Stretch 4 robot.
    """
    base_x: float = 0.0      # Linear velocity along X (m/s)
    base_y: float = 0.0      # Linear velocity along Y (m/s)
    base_theta: float = 0.0  # Angular velocity (rad/s)
    
    lift: float = 0.0        # Prismatic velocity (m/s)
    arm: float = 0.0         # Prismatic velocity (m/s)
    
    wrist_yaw: float = 0.0   # Rotational velocity (rad/s)
    wrist_pitch: float = 0.0 # Rotational velocity (rad/s)
    wrist_roll: float = 0.0  # Rotational velocity (rad/s)

    def get_joint_names(self) -> list[str]:
        """
        Returns the joint names in the order corresponding to the numpy array.
        
        Returns:
            list[str]: The joint names.
        """
        return [
            "base_x",
            "base_y",
            "base_theta",
            "lift",
            "arm",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        ]

    def to_numpy(self) -> np.ndarray:
        """
        Converts the joint velocities to a standard 8-element NumPy array.

        Returns:
            np.ndarray: The 8-element joint velocity vector.
        """
        return np.array([
            self.base_x,
            self.base_y,
            self.base_theta,
            self.lift,
            self.arm,
            self.wrist_yaw,
            self.wrist_pitch,
            self.wrist_roll,
        ])

    @classmethod
    def from_numpy(cls, v: np.ndarray) -> "StretchJointVelocities":
        """
        Creates a StretchJointVelocities instance from Pinocchio's 8-element velocity vector.

        Args:
            v (np.ndarray): Pinocchio's 8-element velocity vector.

        Returns:
            StretchJointVelocities: Instance of StretchJointVelocities.
        """
        return cls(
            base_x=v[0],
            base_y=v[1],
            base_theta=v[2],
            lift=v[3],
            arm=v[4],
            wrist_yaw=v[5],
            wrist_pitch=v[6],
            wrist_roll=v[7],
        )

    def pretty_print(self) -> None:
        """
        Pretty-prints each joint name and velocity value on sequential lines.
        """
        for field in fields(self):
            print(f"{field.name}_dot: {getattr(self, field.name):.4f}")


class StretchKinematics:
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Base class for handling the Pinocchio boilerplate for Stretch 4.

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        # use a full DOF model for control and forward kinematics
        self.model = _load_stretch4_urdf(
            ik_mode=Stretch4IKModes.BASE_PLANAR,
            calibrated=use_calibrated_urdf
        )
        self.data = self.model.createData()

        # use the 6-dof model for inverse kinematics (no translation on base)
        self.model_ik = _load_stretch4_urdf(
            ik_mode=Stretch4IKModes.BASE_ROTATE,
            calibrated=use_calibrated_urdf
        )
        self.data_ik = self.model_ik.createData()

        # Define translation and rotation joints
        self.translation_joints = [
            "mobile_base_planar_joint", 
            "lift_joint", 
            "arm_l4_joint"
        ]
        self.translation_joint_ids = [
            self.model.getJointId(n) for n in self.translation_joints \
            if self.model.existJointName(n)
        ]
        
        self.rotation_joints = [
            "wrist_yaw_joint", 
            "wrist_pitch_joint", 
            "wrist_roll_joint"
        ]
        self.rotation_joint_ids = [
            self.model.getJointId(n) for n in self.rotation_joints \
            if self.model.existJointName(n)
        ]

    def forward(self, q_state: StretchJointPositions, target_frame: str) -> pin.SE3:
        """
        Computes the forward kinematics for the given joint configuration.

        Args:
            q_state (StretchJointPositions): The robot's joint configuration.
            target_frame (str): The name of the frame to compute the forward kinematics for.
        
        Returns:
            pin.SE3: The pose of the target frame in the world frame.
        """
        q = q_state.to_pinocchio_q()
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        frame_id = self.model.getFrameId(target_frame)
        return self.data.oMf[frame_id]

    def _closed_loop_inverse_kinematics(
        self,
        model: pin.Model,
        data: pin.Data,
        target_frame: str,
        target_pose: pin.SE3,
        q_guess: np.ndarray = None,
        max_iter: int = 200,
        eps: float = 1e-4,
        damp: float = 1e-6
    ) -> np.ndarray:
        """
        Computes the numerical inverse kinematics using the standard Closed-Loop 
        Inverse Kinematics (CLIK) algorithm with Levenberg-Marquardt damping.
        
        Reference:
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html

        Args:
            target_frame (str): The name of the frame to compute the inverse kinematics for.
            target_pose (pin.SE3): The desired pose of the target frame in the world frame.
            q_guess (np.ndarray, optional): Initial joint configuration guess. Defaults to neutral configuration.
            max_iter (int): Maximum number of iterations.
            eps (float): Convergence tolerance.
            damp (float): Damping factor for pseudo-inverse.

        Returns:
            np.ndarray: The joint configuration solving the IK.
        """
        frame_id = model.getFrameId(target_frame)
        
        # Initialize q_guess
        if q_guess is not None:
            q = q_guess.copy()
        else:
            q = pin.neutral(model)

        # Clip to joint limits
        if model.nq == model.nv:
            q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
        
        # CLIK algorithm
        for i in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            
            dMi = target_pose.actInv(data.oMf[frame_id])
            err = pin.log(dMi).vector
            
            if np.linalg.norm(err) < eps:
                break
            
            J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
            
            J_JT = J @ J.T + damp * np.eye(6)
            dq = -J.T @ np.linalg.solve(J_JT, err)
            
            q = pin.integrate(model, q, dq)
            
            # Clip to joint limits
            if model.nq == model.nv:
                q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

        return q

    def inverse_6dof(
        self,
        target_frame: str,
        target_pose: pin.SE3 = None,
        target_xyz: np.ndarray = None,
        target_quat: np.ndarray = None,
        target_rpy: np.ndarray = None,
        q_guess: np.ndarray = None,
        max_iter: int = 200,
        eps: float = 1e-4,
        damp: float = 1e-6
    ) -> StretchJointPositions:
        """
        Uses the 6-dof URDF model with only a rotating base (no translation) to solve IK.
        Provide either a target_pose, or a target_xyz and target_quat/target_rpy.
        If target_pose is provided, the other arguments will be ignored.

        Args:
            target_frame (str): The name of the frame to compute the inverse kinematics for.
            target_pose (pin.SE3, optional): The desired pose of the target frame in the world frame.
            target_xyz (np.ndarray, optional): The desired position of the target frame in the world frame.
            target_quat (np.ndarray, optional): The desired orientation of the target frame as a quaternion (scalar-last: [x, y, z, w]).
            target_rpy (np.ndarray, optional): The desired orientation of the target frame as RPY angles (radians).
            q_guess (np.ndarray, optional): Initial joint configuration guess. Defaults to neutral configuration.
            max_iter (int, optional): Maximum number of iterations.
            eps (float, optional): Convergence tolerance.
            damp (float, optional): Damping factor for pseudo-inverse.

        Returns:
            StretchJointPositions: The joint configuration solving the IK.
        """
        # Parse coordinates into a pin.SE3 object
        if target_pose is None:
            if target_xyz is None:
                raise ValueError("Must specify either 'target_pose' or 'target_xyz' position.")
            
            # Position
            translation = np.array(target_xyz, dtype=float)
            
            # Orientation
            if target_quat is not None:
                # Normalize quaternion to prevent numerical issues
                q_xyzw = np.array(target_quat, dtype=float)
                q_xyzw /= np.linalg.norm(q_xyzw)
                # Pinocchio Quaternion constructor signature: Quaternion(w, x, y, z)
                rotation = pin.Quaternion(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]).matrix()
            elif target_rpy is not None:
                # Convert RPY (Roll-Pitch-Yaw) in radians to a rotation matrix
                rotation = pin.rpy.rpyToMatrix(np.array(target_rpy, dtype=float))
            else:
                # Default to identity rotation if none specified
                rotation = np.eye(3)
                
            target_pose = pin.SE3(rotation, translation)

        # Parse q_guess into the 6-dof format for the solver
        if q_guess is not None:
            if isinstance(q_guess, StretchJointPositions):
                q = q_guess.to_pinocchio_q(Stretch4IKModes.BASE_ROTATE)
            elif len(q_guess) == 8:
                q = np.array([q_guess[2], q_guess[3], q_guess[4], q_guess[5], q_guess[6], q_guess[7]])
            elif len(q_guess) != self.model_ik.nq:
                positions = StretchJointPositions.from_pinocchio_q(q_guess)
                q = positions.to_pinocchio_q(Stretch4IKModes.BASE_ROTATE)
            else:
                q = q_guess.copy()
        else:
            q = pin.neutral(self.model_ik)
        
        # Solve IK using initial guess q
        q_6dof = self._closed_loop_inverse_kinematics(
            model=self.model_ik,
            data=self.data_ik,
            target_frame=target_frame,
            target_pose=target_pose,
            q_guess=q,
            max_iter=max_iter,
            eps=eps,
            damp=damp
        )

        solution_pose = self.forward(StretchJointPositions.from_pinocchio_q(q_6dof), target_frame)
        error = np.linalg.norm(pin.log(solution_pose.actInv(target_pose)).vector)
        if error > eps:
            print(f"Warning: IK solution error is {error} with initial guess {q_guess}.")
            print(f"Retrying with neutral pose initial guess.")

            q_6dof = self._closed_loop_inverse_kinematics(
                model=self.model_ik,
                data=self.data_ik,
                target_frame=target_frame,
                target_pose=target_pose,
                q_guess=pin.neutral(self.model_ik),
                max_iter=max_iter,
                eps=eps,
                damp=damp
            )
            solution_pose = self.forward(StretchJointPositions.from_pinocchio_q(q_6dof), target_frame)
            error = np.linalg.norm(pin.log(solution_pose.actInv(target_pose)).vector)
            if error > eps:
                print(f"Warning: IK solution error is {error} with neutral initial guess.")
                raise ValueError("IK solution error is too large. Try a different initial guess or target pose.")
            else:
                print(f"Neutral pose initial guess worked!")
        
        # Map 6-dof solved state back to the 8-joint StretchJointPositions
        return StretchJointPositions.from_pinocchio_q(q_6dof)

    def differential_ik(
        self,
        q: StretchJointPositions,
        target_frame: str,
        v_desired: np.ndarray,
    ) -> StretchJointVelocities:
        """
        Abstract method to be implemented depending on the specific control mode / kinematic structure.

        Computes the joint velocities required to achieve the desired Cartesian velocity.

        Args:
            q (StretchJointPositions): The robot's joint configuration.
            target_frame (str): The name of the frame to compute the velocity relationship for.
            v_desired (np.ndarray): The desired velocity of the target frame (translation and rotation).
                                    Vector of len 6, where the first 3 elements are the linear velocity
                                    and the last 3 elements are the angular velocity.
        
        Returns:
            StretchJointVelocities: Joint velocities required to achieve the target velocity.
        """
        raise NotImplementedError


class ToolFrameKinematics(StretchKinematics):
    # old "mode 1"
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Kinematic model for the local "Tool Frame" relative control mode.
        Provides Cartesian translation relative to the robot's local tool plate / 
        gripper coordinate system (X=forward along the gripper, Y=left, Z=up).

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        super().__init__(use_calibrated_urdf)

        # configuration for weighted-least-squares damping
        self._jacobian_weights = np.array([
            50.0,  # Base X (discouraged)
            50.0,  # Base Y (discouraged)
            1.0,   # Base Theta (encouraged)
            1.0,   # Lift (encouraged)
            1.0    # Arm (encouraged)
        ])
        
        # Scaling factor for the Jacobian weights when a joint limit is approached.
        self._jacobian_weight_scaler_joint_limits = 1000.0

        damping_coefficient = 1.e-4
        self._damping_matrix = np.eye(3) * damping_coefficient

        # Extract columns corresponding to the 5 translational DOFs for Jacobian math
        self.translation_joint_col_idx = []
        for j_id in self.translation_joint_ids:
            idx_v = self.model.joints[j_id].idx_v
            nv = self.model.joints[j_id].nv
            self.translation_joint_col_idx.extend(range(idx_v, idx_v + nv))

    def _compute_constrained_jacobian_weights(
        self,
        q: StretchJointPositions,
        dq_candidate: np.ndarray,
        lift_blend_margin_meters: float = 0.20,
        arm_blend_margin_extension: float = 0.20,
        arm_blend_power_extension: float = 2.0,
        arm_blend_margin_retraction: float = 0.05,
    ) -> np.ndarray:
        """
        Computes constrained Jacobian pinv weights based on the current joint configuration.
        
        Based on active set enforcement / continuous redundancy resolution.

        References:
            - TODO
        
        Args:
            q (StretchJointPositions): The robot's joint configuration.
            dq_candidate (np.ndarray): Candidate joint velocities (5-DOF).
            lift_blend_margin_meters (float): Distance in meters from the lift joint limits to start blending.
            arm_blend_margin_extension (float): Distance in meters from the arm extension limit to start blending.
            arm_blend_power_extension (float): Power to scale the arm blend penalty.
            arm_blend_margin_retraction (float): Distance in meters from the arm retraction limit to start blending.
        
        Returns:
            np.ndarray: The constrained Jacobian pinv weights.
        """
        # Get initial weights from model config
        W = self._jacobian_weights.copy()

        # Get joint position limit indices from model
        lift_j_id = self.model.getJointId("lift_joint")
        arm_j_id = self.model.getJointId("arm_l4_joint")

        lift_idx_q = self.model.joints[lift_j_id].idx_q
        arm_idx_q = self.model.joints[arm_j_id].idx_q

        lift_lower = self.model.lowerPositionLimit[lift_idx_q]
        lift_upper = self.model.upperPositionLimit[lift_idx_q]
        arm_lower = self.model.lowerPositionLimit[arm_idx_q]
        arm_upper = self.model.upperPositionLimit[arm_idx_q]

        # 1. Lift joint limit check (col index 3)
        ratio_lift = _compute_limit_interpolation_ratio(
            q.lift, lift_lower, lift_upper, dq_candidate[3],
            lift_blend_margin_meters, lift_blend_margin_meters
        )
        if ratio_lift > 0.0:
            W[3] = 1.0 + (self._jacobian_weight_scaler_joint_limits - 1.0) * ratio_lift

        # 2. Arm joint limit check (col index 4)
        ratio_arm = _compute_limit_interpolation_ratio(
            q.arm, arm_lower, arm_upper, dq_candidate[4],
            arm_blend_margin_retraction, arm_blend_margin_extension
        )
        if ratio_arm > 0.0:
            # Apply a higher penalty for violating the arm joint limit
            # This makes the arm more "stiff" when it's close to its limits
            ratio_arm = ratio_arm ** arm_blend_power_extension

            W[4] = 1.0 + (self._jacobian_weight_scaler_joint_limits - 1.0) * ratio_arm

        return W

    def differential_ik(self, q: StretchJointPositions, target_frame: str, v_desired: np.ndarray) -> StretchJointVelocities:
        """
        Computes the joint velocities required to achieve the desired Cartesian velocity
        in the tool frame.

        Uses a weighted pseudoinverse to solve for the joint velocities, preferring to 
        use the base rotation and lift DOFs over base translation when possible.

        Args:
            q (StretchJointPositions): The robot's joint configuration.
            target_frame (str): The name of the frame to compute the velocity relationship for.
            v_desired (np.ndarray): The desired 6D twist of the target frame.
                                    The linear velocity component (first 3 elements) is used.
        
        Returns:
            StretchJointVelocities: Joint velocities required to achieve the target velocity.
        """
        q_pin = q.to_pinocchio_q()

        # Jacobian in the gripper's LOCAL frame
        J_full = pin.computeFrameJacobian(
            self.model,
            self.data,
            q_pin,
            self.model.getFrameId(target_frame),
            pin.ReferenceFrame.LOCAL
        )
        
        # Extract linear velocity components
        # Local axes: X (forward), Y (left), Z (up)
        v_fwd_row = J_full[0, :]
        v_left_row = J_full[1, :]
        v_up_row = J_full[2, :]
    
        # We want the output vector to align with: [v_forward, v_left, v_up]
        J_mode1_full = np.vstack([v_fwd_row, v_left_row, v_up_row])

        # truncate Jacobian to only translational DOFs
        J_mode1_trans = J_mode1_full[:, self.translation_joint_col_idx]

        # Extract linear velocity components from 6D desired twist
        v_linear = v_desired[:3]

        # First pass: compute weighted least squares solution with initial weights
        W_pinv = np.diag(1.0 / self._jacobian_weights)
        J_W_JT = J_mode1_trans @ W_pinv @ J_mode1_trans.T
        J_W_JT_damped = J_W_JT + self._damping_matrix
        J_pinv = W_pinv @ J_mode1_trans.T @ np.linalg.inv(J_W_JT_damped)
        dq = J_pinv @ v_linear

        # Second pass: compute active set weights and resolve if weights change
        W = self._compute_constrained_jacobian_weights(q, dq)
        if not np.array_equal(W, self._jacobian_weights):
            W_pinv = np.diag(1.0 / W)
            J_W_JT = J_mode1_trans @ W_pinv @ J_mode1_trans.T
            J_W_JT_damped = J_W_JT + self._damping_matrix
            J_pinv = W_pinv @ J_mode1_trans.T @ np.linalg.inv(J_W_JT_damped)
            dq = J_pinv @ v_linear

        # Map dq back to the full joint velocity space (model.nv)
        v_full = np.zeros(self.model.nv)
        v_full[self.translation_joint_col_idx] = dq

        # Convert back to velocity wrapper for return
        v = StretchJointVelocities.from_numpy(v_full)

        # Apply direct wrist yaw compensation instead of including within Jacobian
        # This allows the differential IK to focus on positioning the tool frame
        # without having to tune damping / weights to include rotation
        v.wrist_yaw = -v.base_theta

        return v


class PlanarToolFrameKinematics(StretchKinematics):
    # old "mode 2"
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Kinematic model for the gravity-aligned "Planar Tool Frame" control mode.
        Projects the tool's X (forward) and Y (left) axes onto the horizontal ground plane 
        while aligning the Z axis with gravity. This enables camera-intuitive planar 
        manipulation, preventing the tool from digging into the floor or lifting up when pitched.

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        super().__init__(use_calibrated_urdf)


class CylindricalToolFrameKinematics(StretchKinematics):
    # old "mode 4"
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Kinematic model for the constrained "Cylindrical" control mode.
        Locks base translation (X/Y) and all wrist orientation joints (yaw, pitch, roll). 
        Resolves task-space translation at the grasp center solely using base rotation (theta), 
        lift (Z), and telescoping arm extension (radius), tracking a cylindrical coordinate grid.

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        super().__init__(use_calibrated_urdf)


