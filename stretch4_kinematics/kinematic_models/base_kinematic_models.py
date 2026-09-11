import numpy as np
import pinocchio as pin

from stretch4_kinematics.state.joint_positions import StretchJointPositions
from stretch4_kinematics.state.joint_velocities import StretchJointVelocities
from stretch4_kinematics.state.modes import Stretch4IKModes
from stretch4_kinematics.kinematic_models.utils import _load_stretch4_urdf

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

    def forward_velocity(self, q_state: StretchJointPositions, q_dot_state: StretchJointVelocities, target_frame: str) -> np.ndarray:
        """
        Computes the forward velocity of the specified target frame given the joint velocities.

        Args:
            q_state (StretchJointPositions): The robot's joint configuration.
            q_dot_state (StretchJointVelocities): The joint velocities.
            target_frame (str): The name of the frame to compute the forward velocity for.

        Returns:
            np.ndarray: The spatial velocity of the target frame.
        """
        frame_id = self.model.getFrameId(target_frame)
        q = q_state.to_pinocchio_q()
        J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL)
        v = J @ q_dot_state.to_numpy()
        return v

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

    def inverse_6dof_local(
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
        Solves IK relative the robot's current (local) base position: [X,Y,A] = [0,0,0]
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
                NOTE: q_guess ignores the base [X,Y,A] state
            max_iter (int, optional): Maximum number of iterations.
            eps (float, optional): Convergence tolerance.
            damp (float, optional): Damping factor for pseudo-inverse.

        Returns:
            StretchJointPositions: The joint configuration solving the IK in a local base frame.
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

        # Ensure the solver starts at local base rotation A = 0
        q[0] = 0.0
        
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
        
        # Check if solution converged and retry using pin.neutral as initial guess
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
