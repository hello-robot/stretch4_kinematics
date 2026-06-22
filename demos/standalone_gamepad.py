#!/usr/bin/env python3
import threading
import time
from select import select
import glob

# Try importing evdev, if missing alert user
try:
    from evdev import InputDevice, ecodes, list_devices
except ImportError:
    print("Please install evdev: pip install evdev")
    raise

# --- Utilities to discover a joystick device --------------------------------
WANTED_KEY_CODES = {
    ecodes.BTN_SOUTH, ecodes.BTN_EAST, ecodes.BTN_NORTH, ecodes.BTN_WEST,
    ecodes.BTN_TL, ecodes.BTN_TR, ecodes.BTN_THUMBL, ecodes.BTN_THUMBR,
    ecodes.BTN_SELECT, ecodes.BTN_START, getattr(ecodes, "BTN_MODE", 0)
}
WANTED_ABS_CODES = {
    ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_RX, ecodes.ABS_RY,
    ecodes.ABS_Z, ecodes.ABS_RZ, ecodes.ABS_HAT0X, ecodes.ABS_HAT0Y
}

def is_probable_gamepad(dev) -> bool:
    caps = dev.capabilities()
    if ecodes.EV_ABS not in caps or ecodes.EV_KEY not in caps:
        return False
    keyset = set(caps.get(ecodes.EV_KEY, []))
    absset = set(a if isinstance(a, int) else a[0] for a in caps.get(ecodes.EV_ABS, []))
    return (len(keyset & WANTED_KEY_CODES) >= 1) and (len(absset & WANTED_ABS_CODES) >= 2)

def find_first_gamepad():
    for path in sorted(glob.glob("/dev/input/by-id/*-event-joystick")):
        try:
            dev = InputDevice(path)
            ok = is_probable_gamepad(dev)
            dev.close()
            if ok: return path
        except Exception: pass
    for path in list_devices():
        try:
            dev = InputDevice(path)
            ok = is_probable_gamepad(dev)
            dev.close()
            if ok: return path
        except Exception: pass
    return None

class UnpluggedError(Exception): pass

class GPEvent:
    def __init__(self, code, state, ev_type=""):
        self.code = code
        self.state = state
        self.ev_type = ev_type

class Stick:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.norm = float(pow(2, 15))
    def update_x(self, abs_x): self.x = int(abs_x) / self.norm
    def update_y(self, abs_y): self.y = -int(abs_y) / self.norm

class Button:
    def __init__(self): self.pressed = False
    def update(self, state): self.pressed = (state == 1)

class Trigger:
    def __init__(self, xbox_one=False):
        num_bits = 10 if xbox_one else 8
        self.norm = float(pow(2, num_bits) - 1)
        self.pulled = 0.0
    def update(self, state):
        self.pulled = int(state) / self.norm
        if self.pulled > 1.0: self.pulled = 1.0

class StandaloneGamePadController:
    def __init__(self, print_events=False, print_dongle_status=True, is_xbox_one=False):
        self.print_events = print_events
        self.print_dongle_status = print_dongle_status
        self.left_stick = Stick()
        self.right_stick = Stick()
        self.left_stick_button = Button()
        self.right_stick_button = Button()
        self.middle_led_ring_button = Button()
        self.bottom_button = Button()
        self.top_button = Button()
        self.left_button = Button()
        self.right_button = Button()
        self.right_shoulder_button = Button()
        self.left_shoulder_button = Button()
        self.select_button = Button()
        self.start_button = Button()
        self.left_trigger = Trigger(xbox_one=is_xbox_one)
        self.right_trigger = Trigger(xbox_one=is_xbox_one)
        self.left_pad = Button()
        self.right_pad = Button()
        self.top_pad = Button()
        self.bottom_pad = Button()
        self.lock = threading.Lock()
        self.device_path = None
        self.dev = None
        self.is_gamepad_active = False
        self.last_event_ts = 0.0
        self.EVENT_ACTIVITY_TIMEOUT = 0.5
        self.zero_state_sent_counter = 6
        self.STOP_FRAME_COUNT = 5
        self.thread = None
        self.thread_shutdown_flag = threading.Event()

    def startup(self):
        if self.thread is not None:
            self.thread_shutdown_flag.set()
            self.thread.join(1)
        self.thread = threading.Thread(target=self._thread_target)
        self.thread.daemon = True
        self.thread_shutdown_flag.clear()
        self.thread.start()
        return True

    def _thread_target(self):
        while not self.thread_shutdown_flag.is_set():
            self.update()
            time.sleep(0.04) # roughly 25Hz

    def stop(self):
        self.thread_shutdown_flag.set()
        if self.thread is not None:
            self.thread.join(1)
        if self.dev is not None:
            try: self.dev.close()
            except: pass

    def poll_till_gamepad_dongle_present(self):
        with self.lock:
            self.is_gamepad_active = False
        if self.print_dongle_status:
            print("\033[93mWaiting for Gamepad Dongle...\033[0m")
        try:
            self.device_path = find_first_gamepad()
            if self.device_path:
                self.dev = InputDevice(self.device_path)
                print(f"\033[92m\033[1mGamepad Dongle FOUND! ({self.dev.name})\033[0m")
                with self.lock:
                    self.is_gamepad_active = True
        except Exception:
            pass

    def get_gamepad_events(self):
        if not self.dev:
            raise UnpluggedError("No gamepad found.")
        r, _, _ = select([self.dev.fd], [], [], 0.02)
        if not r: return []
        events = []
        try:
            for ev in self.dev.read():
                if ev.type not in (ecodes.EV_KEY, ecodes.EV_ABS): continue
                code_name = ecodes.bytype[ev.type][ev.code]
                events.append(GPEvent(code=code_name, state=ev.value, ev_type="EV_KEY" if ev.type == ecodes.EV_KEY else "EV_ABS"))
        except BlockingIOError: pass
        except OSError: raise UnpluggedError("Gamepad disconnected.")
        return events

    def update(self):
        if not self.is_gamepad_active:
            self.poll_till_gamepad_dongle_present()
            return
        try:
            events = self.get_gamepad_events()
            if events: self.last_event_ts = time.monotonic()
            self.update_button_encodings(events)
        except (UnpluggedError, OSError):
            print("\033[91m\033[1mGamepad Dongle DISCONNECTED...\033[0m")
            try: self.dev.close()
            except Exception: pass
            self.dev = None
            with self.lock: self.is_gamepad_active = False
            self.set_zero_state()

    def update_button_encodings(self, events):
        with self.lock:
            for event in events:
                if event.code == 'ABS_X': self.left_stick.update_x(event.state)
                if event.code == 'ABS_Y': self.left_stick.update_y(event.state)
                if event.code == 'ABS_RX': self.right_stick.update_x(event.state)
                if event.code == 'ABS_RY': self.right_stick.update_y(event.state)
                if event.code == 'BTN_MODE': self.middle_led_ring_button.update(event.state)
                if 'BTN_SOUTH' in list(event.code): self.bottom_button.update(event.state)
                if 'BTN_WEST' in list(event.code): self.top_button.update(event.state)
                if 'BTN_NORTH' in list(event.code): self.left_button.update(event.state)
                if 'BTN_EAST' in list(event.code): self.right_button.update(event.state)
                if event.code == 'BTN_TL': self.left_shoulder_button.update(event.state)
                if event.code == 'BTN_TR': self.right_shoulder_button.update(event.state)
                if event.code == 'ABS_Z': self.left_trigger.update(event.state)
                if event.code == 'ABS_RZ': self.right_trigger.update(event.state)
                if event.code == 'BTN_SELECT': self.select_button.update(event.state)
                if event.code == 'BTN_START': self.start_button.update(event.state)
                if event.code == 'BTN_THUMBL': self.left_stick_button.update(event.state)
                if event.code == 'BTN_THUMBR': self.right_stick_button.update(event.state)
                if event.code == 'ABS_HAT0Y':
                    if event.state == 0: self.top_pad.update(0); self.bottom_pad.update(0)
                    elif event.state == 1: self.top_pad.update(0); self.bottom_pad.update(1)
                    elif event.state == -1: self.bottom_pad.update(0); self.top_pad.update(1)
                if event.code == 'ABS_HAT0X':
                    if event.state == 0: self.left_pad.update(0); self.right_pad.update(0)
                    elif event.state == 1: self.left_pad.update(0); self.right_pad.update(1)
                    elif event.state == -1: self.right_pad.update(0); self.left_pad.update(1)
                if self.print_events: print(event.ev_type, event.code, event.state)

    def set_zero_state(self):
        with self.lock:
            self.middle_led_ring_button.pressed = False
            self.left_stick.x = 0; self.left_stick.y = 0
            self.right_stick.x = 0; self.right_stick.y = 0
            self.left_stick_button.pressed = False; self.right_stick_button.pressed = False
            self.bottom_button.pressed = False; self.top_button.pressed = False
            self.left_button.pressed = False; self.right_button.pressed = False
            self.left_shoulder_button.pressed = False; self.right_shoulder_button.pressed = False
            self.select_button.pressed = False; self.start_button.pressed = False
            self.bottom_pad.pressed = False; self.top_pad.pressed = False
            self.left_pad.pressed = False; self.right_pad.pressed = False
            self.left_trigger.pulled = 0; self.right_trigger.pulled = 0
        self.zero_state_sent_counter = 0

    def get_state(self):
        with self.lock:
            state = {
                'middle_led_ring_button_pressed': self.middle_led_ring_button.pressed,
                'left_stick_x': self.left_stick.x, 'left_stick_y': self.left_stick.y,
                'right_stick_x': self.right_stick.x, 'right_stick_y': self.right_stick.y,
                'left_stick_button_pressed': self.left_stick_button.pressed,
                'right_stick_button_pressed': self.right_stick_button.pressed,
                'bottom_button_pressed': self.bottom_button.pressed,
                'top_button_pressed': self.top_button.pressed,
                'left_button_pressed': self.left_button.pressed,
                'right_button_pressed': self.right_button.pressed,
                'left_shoulder_button_pressed': self.left_shoulder_button.pressed,
                'right_shoulder_button_pressed': self.right_shoulder_button.pressed,
                'select_button_pressed': self.select_button.pressed,
                'start_button_pressed': self.start_button.pressed,
                'left_trigger_pulled': self.left_trigger.pulled,
                'right_trigger_pulled': self.right_trigger.pulled,
                'bottom_pad_pressed': self.bottom_pad.pressed,
                'top_pad_pressed': self.top_pad.pressed,
                'left_pad_pressed': self.left_pad.pressed,
                'right_pad_pressed': self.right_pad.pressed
            }
            is_active = False
            if time.monotonic() - self.last_event_ts < self.EVENT_ACTIVITY_TIMEOUT: is_active = True
            if not is_active:
                if any([
                    state['middle_led_ring_button_pressed'], state['left_stick_button_pressed'], state['right_stick_button_pressed'],
                    state['bottom_button_pressed'], state['top_button_pressed'], state['left_button_pressed'], state['right_button_pressed'],
                    state['left_shoulder_button_pressed'], state['right_shoulder_button_pressed'], state['select_button_pressed'],
                    state['start_button_pressed'], state['bottom_pad_pressed'], state['top_pad_pressed'], state['left_pad_pressed'], state['right_pad_pressed']
                ]): is_active = True
                elif any(abs(v) > 1e-3 for v in [state['left_stick_x'], state['left_stick_y'], state['right_stick_x'], state['right_stick_y']]): is_active = True
                elif any(v > 1e-3 for v in [state['left_trigger_pulled'], state['right_trigger_pulled']]): is_active = True

            if is_active:
                self.zero_state_sent_counter = 0
                return state
            else:
                if self.zero_state_sent_counter < self.STOP_FRAME_COUNT:
                    self.zero_state_sent_counter += 1
                    return state
                else:
                    return None
        return state
