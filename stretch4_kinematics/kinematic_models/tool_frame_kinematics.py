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

    def diff_ik_testbed(self, q: StretchJointPositions):
        ## 6x8 solution
        q_pin = q.to_pinocchio_q()

        J_full = pin.computeFrameJacobian(
            self.model,
            self.data,
            q_pin,
            self.model.getFrameId("tool_attachment_site_link"),
            pin.ReferenceFrame.LOCAL
        )

        print(np.round(J_full, 3))

        ## SVD
        U, S, Vh = np.linalg.svd(J_full)
        print("U:\n", np.round(U, 3))
        print("S:\n", np.round(S, 3))
        print("Vh:\n", np.round(Vh, 3))

        ## rank
        rank = np.linalg.matrix_rank(J_full)
        print("Rank of J_full:", rank)

        ## orthonormal basis of nullspace
        nullspace_basis = Vh[rank:].T
        print("Orthonormal basis of nullspace:\n", np.round(nullspace_basis, 3))

        print("\n"*3)

        ## nullspace projection operator
        n_dim = self.model.nv
        J_pinv = np.linalg.pinv(J_full)
        P = np.eye(n_dim) - J_pinv @ J_full
        print("Nullspace projection operator P:\n", np.round(P, 3))

        q_0 = np.zeros(self.model.nv)
        q_0[1] = 0.1  # Example initialization for the second joint
        q_null = P @ q_0
        print("Projected q_0 into nullspace:\n", np.round(q_null, 3))

        ## TODO: 3x5 solution using reduced Jacobian in differential_ik

    def nullspace_projection(self, q: StretchJointPositions, q_dot: StretchJointVelocities) -> np.ndarray:
        v_full = q_dot.to_numpy()
        J_full = pin.computeFrameJacobian(
            self.model,
            self.data,
            q.to_pinocchio_q(),
            self.model.getFrameId("tool_attachment_site_link"),
            pin.ReferenceFrame.LOCAL
        )
        J_pinv = np.linalg.pinv(J_full)
        P = np.eye(self.model.nv) - J_pinv @ J_full
        return P @ v_full


