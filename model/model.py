import os
import time
import ctypes
import urllib.request

import cv2
import numpy as np
from pynput.keyboard import Controller as KBController, Key, KeyCode
from tensorflow.lite.python.interpreter import Interpreter

# —— Prevent OpenCV Alt+Tab crashes ——  
cv2.startWindowThread()

# —— Download & load MoveNet Lightning int8 TFLite ——  
MODEL_NAME = "movenet_lightning_int8.tflite"
MODEL_URL = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/"
    "int8/4?lite-format=tflite"
)
if not os.path.exists(MODEL_NAME):
    print("Downloading MoveNet model…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)

interp = Interpreter(model_path=MODEL_NAME)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
INPUT_SIZE = inp_det['shape'][1]

# —— Key mappings & state ——  
keyboard   = KBController()
KEY_W      = KeyCode.from_char('w')
KEY_CROUCH = Key.ctrl
KEY_JUMP   = Key.space

state = {
    'walk':   False,
    'crouch': False,
    'jump':   False,
}

# —— EMA smoothing factor ——  
ALPHA = 0.4
prev_pos = None

# —— Thresholds ——  
CROUCH_TH         = 0.6    # hips normalized y > this → crouch
JUMP_WRIST_THRESH = 0.0    # not used directly; we compare wrists vs head
LEG_LIFT_MARGIN   = 0.02   # ankle must be this much above knee to count as lifted
LOOK_DEADZONE     = 0.05   # ignore small head turns
LOOK_SCALE        = 35     # pixels per fraction-of-screen head turn

# —— Pose indices (COCO ordering) ——  
IDX_NOSE    = 0
IDX_L_EYE   = 1
IDX_R_EYE   = 2
IDX_L_EAR   = 3
IDX_R_EAR   = 4
IDX_L_SHOUL = 5
IDX_R_SHOUL = 6
IDX_L_ELBOW = 7
IDX_R_ELBOW = 8
IDX_L_WRIST = 9
IDX_R_WRIST = 10
IDX_L_HIP   = 11
IDX_R_HIP   = 12
IDX_L_KNEE  = 13
IDX_R_KNEE  = 14
IDX_L_ANKLE = 15
IDX_R_ANKLE = 16

# —— Win32 raw mouse injection ——  
MOUSEEVENTF_MOVE = 0x0001
def send_mouse(dx, dy):
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)

# —— Safe input wrappers ——  
def safe_press(key):
    try:
        keyboard.press(key)
    except Exception as e:
        print("⚠️ press failed:", e)

def safe_release(key):
    try:
        keyboard.release(key)
    except Exception as e:
        print("⚠️ release failed:", e)

def safe_mouse(dx, dy):
    try:
        send_mouse(dx, dy)
    except Exception as e:
        print("⚠️ mouse move failed:", e)

# —— Open webcam and window ——  
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")
cv2.namedWindow("CV → Minecraft", cv2.WINDOW_NORMAL)

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # mirror & dimensions
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # preprocess for MoveNet
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    inp = np.expand_dims(img, 0).astype(np.uint8)
    interp.set_tensor(inp_det['index'], inp)
    interp.invoke()
    kps = interp.get_tensor(out_det['index'])[0, 0, :, :]  # shape (17,3)

    # extract & smooth
    pos  = kps[:, :2]   # y, x normalized
    conf = kps[:,  2]
    if prev_pos is None:
        smooth = pos.copy()
        old    = pos.copy()
    else:
        old    = prev_pos.copy()
        smooth = ALPHA * pos + (1 - ALPHA) * prev_pos

    prev_pos = smooth.copy()

    # —— 1) CROUCH: hold CTRL when hips drop ——  
    hip_y = (smooth[IDX_L_HIP,0] + smooth[IDX_R_HIP,0]) / 2
    crouch = hip_y > CROUCH_TH
    if crouch != state['crouch']:
        if crouch:
            safe_press(KEY_CROUCH)
        else:
            safe_release(KEY_CROUCH)
    state['crouch'] = crouch

    # —— 2) JUMP: tap SPACE when both wrists above head ——  
    head_y = smooth[IDX_NOSE,0]
    lw_y   = smooth[IDX_L_WRIST,0]
    rw_y   = smooth[IDX_R_WRIST,0]
    jump   = (lw_y < head_y) and (rw_y < head_y)
    if jump and not state['jump']:
        safe_press(KEY_JUMP)
        safe_release(KEY_JUMP)
    state['jump'] = jump

    # —— 3) WALK: hold W when either ankle lifts above its knee ——  
    la_y, _ = smooth[IDX_L_ANKLE]
    ra_y, _ = smooth[IDX_R_ANKLE]
    lk_y, _ = smooth[IDX_L_KNEE]
    rk_y, _ = smooth[IDX_R_KNEE]
    leg_lift = ((lk_y - la_y) > LEG_LIFT_MARGIN) or ((rk_y - ra_y) > LEG_LIFT_MARGIN)
    walk     = leg_lift
    if walk != state['walk']:
        if walk:
            safe_press(KEY_W)
        else:
            safe_release(KEY_W)
    state['walk'] = walk

    # —— 4) LOOK: raw mouse move on head turn ——  
    x_head = smooth[IDX_NOSE,1]
    center = 0.5
    dx = 0
    if x_head < center - LOOK_DEADZONE:
        dx = -int( LOOK_SCALE * ((center - LOOK_DEADZONE) - x_head) )
    elif x_head > center + LOOK_DEADZONE:
        dx =  int( LOOK_SCALE * (x_head - (center + LOOK_DEADZONE)) )
    if dx != 0:
        safe_mouse(dx, 0)

    # —— Optionally draw skeleton for debugging ——  
    for i in range(17):
        if conf[i] > 0.3:
            y, x = smooth[i]
            cv2.circle(frame, (int(x*W), int(y*H)), 4, (0,255,0), -1)

    # show FPS
    fps = 1.0 / (time.time() - t0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("CV → Minecraft", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
