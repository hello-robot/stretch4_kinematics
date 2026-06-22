from stretch4_kinematics.kinematic_models.base_kinematic_models import StretchKinematics

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
