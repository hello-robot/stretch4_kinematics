import time
from standalone_gamepad import StandaloneGamePadController

class GamepadMapper:
    def __init__(self):
        # We start the gamepad thread
        # Setting print_dongle_status=True logs the connectivity
        self.controller = StandaloneGamePadController(print_dongle_status=True)
        self.controller.startup()
        
        self.last_toggle_time = 0.0
        self.last_a_time = 0.0
        self.last_b_time = 0.0
        
    def get_commands(self):
        """
        Polls the gamepad state dictionary mapped by GamePadController
        and returns exactly the layout expected by the visualization IK script.
        """
        state = self.controller.get_state()
        if not state:
            return None
            
        def deadzone(val, thresh=0.15):
            return val if abs(val) > thresh else 0.0
            
        v_desired = [0.0, 0.0, 0.0]
        rot_change = [0.0, 0.0, 0.0]
        
        # -----------------------------------
        # Left Stick: Forward / Left
        # -----------------------------------
        # Left stick Y (up is positive) -> v_fwd (v_desired[0])
        v_desired[0] = deadzone(state.get('left_stick_y', 0.0))
        # Left stick X (right is positive) -> v_left (v_desired[1] = -X)
        v_desired[1] = deadzone(-state.get('left_stick_x', 0.0))
        
        # -----------------------------------
        # Right Stick: Pitch / Yaw
        # -----------------------------------
        # Right stick Y (up is positive) -> Pitch Up (rot_change[1])
        rot_change[1] = deadzone(state.get('right_stick_y', 0.0))
        # Right stick X (right is positive) -> Yaw Left (rot_change[0] = -X)
        rot_change[0] = deadzone(-state.get('right_stick_x', 0.0))
        
        # -----------------------------------
        # D-pad: Up / Roll
        # -----------------------------------
        if state.get('top_pad_pressed'): v_desired[2] = 1.0
        elif state.get('bottom_pad_pressed'): v_desired[2] = -1.0
        
        if state.get('left_pad_pressed'): rot_change[2] = -1.0
        elif state.get('right_pad_pressed'): rot_change[2] = 1.0
        
        # -----------------------------------
        # Buttons: Mode Toggle & Gripper
        # -----------------------------------
        # Y button (top_button) maps to toggle modes
        toggle = False
        if state.get('top_button_pressed'):
            if time.time() - self.last_toggle_time > 0.5:
                toggle = True
                self.last_toggle_time = time.time()
                
        # A button (bottom_button) = Close
        # B button (right_button) = Open
        grip_cmd = None
        if state.get('bottom_button_pressed'):
            if time.time() - self.last_a_time > 0.2:
                grip_cmd = "CLOSE"
                self.last_a_time = time.time()
        elif state.get('right_button_pressed'):
            if time.time() - self.last_b_time > 0.2:
                grip_cmd = "OPEN"
                self.last_b_time = time.time()
                
        return {
            'v_desired': v_desired,
            'rot_change': rot_change,
            'toggle': toggle,
            'grip': grip_cmd,
            'left_trigger': state.get('left_trigger_pulled', 0.0),
            'right_trigger': state.get('right_trigger_pulled', 0.0)
        }
        
    def stop(self):
        self.controller.stop()
