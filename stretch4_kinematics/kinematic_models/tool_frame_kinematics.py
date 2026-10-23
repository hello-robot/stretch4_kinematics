import numpy as np
import pinocchio as pin

from stretch4_kinematics.state.joint_positions import StretchJointPositions
from stretch4_kinematics.state.joint_velocities import StretchJointVelocities
from stretch4_kinematics.kinematic_models.base_kinematic_models import StretchKinematics
from stretch4_kinematics.kinematic_models.utils import _compute_limit_interpolation_ratio

class ToolFrameKinematics(StretchKinematics):
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Kinematic model for the local "Tool Frame" relative control mode.
        Provides Cartesian translation relative to the robot's local tool plate / 
        gripper coordinate system (X=forward along the gripper, Y=left, Z=up).

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        super().__init__(use_calibrated_urdf)

        # Frame ID lookup cache
        self._frame_id_cache: dict[str, int] = {}

        # Cache joint position limit indices and lower/upper bounds for lift & arm
        self._lift_j_id = self.model.getJointId("lift_joint")
        self._arm_j_id = self.model.getJointId("arm_l4_joint")

        self._lift_idx_q = self.model.joints[self._lift_j_id].idx_q
        self._arm_idx_q = self.model.joints[self._arm_j_id].idx_q

        self._lift_lower = float(self.model.lowerPositionLimit[self._lift_idx_q])
        self._lift_upper = float(self.model.upperPositionLimit[self._lift_idx_q])
        self._arm_lower = float(self.model.lowerPositionLimit[self._arm_idx_q])
        self._arm_upper = float(self.model.upperPositionLimit[self._arm_idx_q])

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

    def _get_frame_id(self, target_frame: str) -> int:
        """
        Get and cache the Pinocchio frame ID for target_frame.

        Args:
            target_frame: Name of the link frame.

        Returns:
            int: Pinocchio frame ID index.
        """
        if target_frame not in self._frame_id_cache:
            self._frame_id_cache[target_frame] = self.model.getFrameId(target_frame)
        return self._frame_id_cache[target_frame]

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

        # 1. Lift joint limit check (col index 3)
        ratio_lift = _compute_limit_interpolation_ratio(
            q.lift, self._lift_lower, self._lift_upper, dq_candidate[3],
            lift_blend_margin_meters, lift_blend_margin_meters
        )
        if ratio_lift > 0.0:
            W[3] = 1.0 + (self._jacobian_weight_scaler_joint_limits - 1.0) * ratio_lift

        # 2. Arm joint limit check (col index 4)
        ratio_arm = _compute_limit_interpolation_ratio(
            q.arm, self._arm_lower, self._arm_upper, dq_candidate[4],
            arm_blend_margin_retraction, arm_blend_margin_extension
        )
        if ratio_arm > 0.0:
            # Apply a higher penalty for violating the arm joint limit
            # This makes the arm more "stiff" when it's close to its limits
            ratio_arm = ratio_arm ** arm_blend_power_extension

            W[4] = 1.0 + (self._jacobian_weight_scaler_joint_limits - 1.0) * ratio_arm

        return W

    def _solve_weighted_dls(self, J: np.ndarray, v: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Solves weighted damped least squares without explicit matrix inversion or diagonal matrices.

        Args:
            J: Jacobian matrix of shape (3, 5).
            v: Desired linear velocity vector of shape (3,).
            W: Joint weight vector of shape (5,).

        Returns:
            np.ndarray: Weighted joint velocity solution vector of shape (5,).
        """
        JW_inv = J / W  # Broadcasts 1/W across columns: equivalent to J @ diag(1/W)
        JJT_damped = JW_inv @ J.T + self._damping_matrix
        return JW_inv.T @ np.linalg.solve(JJT_damped, v)

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
            self._get_frame_id(target_frame),
            pin.ReferenceFrame.LOCAL
        )
        
        # Truncate Jacobian to linear components and translational DOFs
        J_mode1_trans = J_full[:3, self.translation_joint_col_idx]
        v_linear = v_desired[:3]

        # First pass: compute weighted least squares solution with initial weights
        dq = self._solve_weighted_dls(J_mode1_trans, v_linear, self._jacobian_weights)

        # Second pass: compute active set weights and resolve if weights change
        W = self._compute_constrained_jacobian_weights(q, dq)
        if not np.array_equal(W, self._jacobian_weights):
            dq = self._solve_weighted_dls(J_mode1_trans, v_linear, W)

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

    def nullspace_projection_joint_space(
        self,
        q: StretchJointPositions,
        q_dot: StretchJointVelocities,
        target_frame: str = "tool_attachment_site_link",
        damping: float = 1e-4,
    ) -> StretchJointVelocities:
        """
        Compute naive joint space velocity projection into the nullspace of a target frame's Jacobian.

        Projects a candidate 8-DOF joint velocity vector into the nullspace matrix
        P = (I - J^+ J) of the primary task frame Jacobian, ensuring the resulting joint
        velocities produce zero spatial velocity at the specified target frame.

        Args:
            q: The robot's current 8-DOF joint configuration.
            q_dot: The candidate 8-DOF joint velocity vector to project.
            target_frame: The name of the kinematic link frame whose end-effector
                velocity should be constrained to zero.
            damping: Regularization factor for the pseudoinverse calculation.

        Returns:
            StretchJointVelocities: The projected 8-DOF joint velocity commands.
        """
        q_pin = q.to_pinocchio_q()
        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            q_pin,
            self._get_frame_id(target_frame),
            pin.ReferenceFrame.LOCAL,
        )
        nv = self.model.nv
        if damping > 0.0:
            J_pinv = np.linalg.solve(J @ J.T + damping * np.eye(6), J).T
        else:
            J_pinv = np.linalg.pinv(J)
        P = np.eye(nv) - J_pinv @ J
        v_proj = P @ q_dot.to_numpy()
        return StretchJointVelocities.from_numpy(v_proj)

    def hierarchical_differential_ik(
        self,
        q: StretchJointPositions,
        v1: np.ndarray,
        J1: np.ndarray | None = None,
        target_frame: str = "tool_attachment_site_link",
        J2: np.ndarray | None = None,
        v2: np.ndarray | None = None,
        damping: float = 1e-3,
    ) -> StretchJointVelocities:
        """
        Compute 8-DOF joint velocities using multi-task prioritized differential IK.

        Solves for joint velocities that satisfy a primary task (J1, v1) while
        attempting to satisfy an arbitrary secondary task (J2, v2) strictly within
        the primary task's nullspace using np.linalg.solve to avoid explicit inversion.

        If J1 is not provided, it is automatically computed as the 6x8 spatial Jacobian
        of target_frame in the local reference frame.

        Task hierarchy equation:
            J1_pinv = (J1 @ J1.T + damping * I)^-1 @ J1
            dq_primary = J1_pinv @ v1
            P1 = I - J1_pinv @ J1
            dq_secondary = P1 @ (J2_tilde.T @ (J2_tilde @ J2_tilde.T + damping * I)^-1 @ (v2 - J2 @ dq_primary))
            dq_total = dq_primary + dq_secondary

        Args:
            q: The robot's current 8-DOF joint configuration.
            v1: Primary task space velocity vector of shape (m1,).
            J1: Optional primary task Jacobian matrix of shape (m1, 8). If None,
                computed as the (6, 8) spatial Jacobian of target_frame.
            target_frame: Kinematic link frame used to compute J1 if J1 is None.
            J2: Optional secondary task Jacobian matrix of shape (m2, 8).
            v2: Optional secondary task space velocity vector of shape (m2,).
            damping: Levenberg-Marquardt damping factor for pseudoinverse calculations.

        Returns:
            StretchJointVelocities: The resulting 8-DOF joint velocity commands.
        """
        nv = self.model.nv  # 8 DOFs

        if J1 is None:
            q_pin = q.to_pinocchio_q()
            J1 = pin.computeFrameJacobian(
                self.model,
                self.data,
                q_pin,
                self._get_frame_id(target_frame),
                pin.ReferenceFrame.LOCAL,
            )

        m1 = J1.shape[0]
        # Solve (J1 @ J1.T + damping * I) * X = J1 for X = J1_pinv.T
        J1_pinv = np.linalg.solve(J1 @ J1.T + damping * np.eye(m1), J1).T
        dq_primary = J1_pinv @ v1

        if J2 is None or v2 is None:
            return StretchJointVelocities.from_numpy(dq_primary)

        P1 = np.eye(nv) - J1_pinv @ J1
        m2 = J2.shape[0]
        J2_tilde = J2 @ P1

        v2_error = v2 - J2 @ dq_primary
        y = np.linalg.solve(J2_tilde @ J2_tilde.T + damping * np.eye(m2), v2_error)
        dq_secondary = P1 @ (J2_tilde.T @ y)

        return StretchJointVelocities.from_numpy(dq_primary + dq_secondary)

    def nullspace_base_control(
        self,
        q: StretchJointPositions,
        v_ee_local_desired: np.ndarray | None = None,
        v_base_world_desired: np.ndarray | None = None,
        target_frame: str = "tool_attachment_site_link",
        damping: float = 1e-3,
    ) -> StretchJointVelocities:
        """
        Compute 8-DOF joint velocities that prioritize end-effector motion while
        driving the mobile base in the world XY plane within the primary task's nullspace.

        Args:
            q: The robot's current 8-DOF joint configuration.
            v_ee_local_desired: Desired 6D end-effector spatial velocity in the local
                tool frame of shape (6,). Defaults to zeros to keep EE stationary.
            v_base_world_desired: Desired 2D linear base velocity [vx_world, vy_world]
                in meters per second in the world frame of shape (2,). Defaults to zeros.
            target_frame: The name of the kinematic link frame for the primary task.
            damping: Levenberg-Marquardt damping factor for the pseudoinverses.

        Returns:
            StretchJointVelocities: The resulting 8-DOF joint velocity commands.
        """
        q_pin = q.to_pinocchio_q()
        nv = self.model.nv

        # Primary Task: 6x8 End-Effector Jacobian (LOCAL frame)
        J1 = pin.computeFrameJacobian(
            self.model,
            self.data,
            q_pin,
            self._get_frame_id(target_frame),
            pin.ReferenceFrame.LOCAL,
        )
        v1 = np.zeros(6) if v_ee_local_desired is None else v_ee_local_desired

        # Secondary Task: 2x8 Base XY Jacobian in World Frame.
        # Rotate local base tangent velocities by base_theta to map to world XY.
        theta = q.base_theta
        J2 = np.zeros((2, nv))
        J2[0, 0], J2[0, 1] = np.cos(theta), -np.sin(theta)
        J2[1, 0], J2[1, 1] = np.sin(theta), np.cos(theta)
        v2 = np.zeros(2) if v_base_world_desired is None else v_base_world_desired

        return self.hierarchical_differential_ik(
            q=q,
            v1=v1,
            J1=J1,
            target_frame=target_frame,
            J2=J2,
            v2=v2,
            damping=damping,
        )
