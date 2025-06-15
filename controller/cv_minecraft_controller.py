import os
import time
import ctypes
import urllib.request

import cv2
import numpy as np
from pynput.keyboard import Controller as KBController, Key, KeyCode
from tensorflow.lite.python.interpreter import Interpreter

# —— Ensure HighGUI window thread is running ——
cv2.startWindowThread()

# —— Download & load MoveNet model ——
MODEL_NAME = "movenet_lightning_int8.tflite"
MODEL_URL  = (
    "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/"
    "int8/4?lite-format=tflite"
)
if not os.path.exists(MODEL_NAME):
    urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)

interpreter = Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()
inp_det  = interpreter.get_input_details()[0]
out_det  = interpreter.get_output_details()[0]
INPUT_SZ = inp_det['shape'][1]

# —— Keyboard & mouse ——
keyboard    = KBController()
KEY_W       = KeyCode.from_char('w')
KEY_CROUCH  = Key.ctrl
KEY_JUMP    = Key.space
# mouse click flags
def click_left():
    ctypes.windll.user32.mouse_event(0x0002,0,0,0,0)
    ctypes.windll.user32.mouse_event(0x0004,0,0,0,0)
def click_right():
    ctypes.windll.user32.mouse_event(0x0008,0,0,0,0)
    ctypes.windll.user32.mouse_event(0x0010,0,0,0,0)

def send_mouse(dx, dy):
    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)

def safe_press(k): keyboard.press(k)
def safe_release(k): keyboard.release(k)
def safe_mouse(dx, dy): send_mouse(dx, dy)

# —— State ——
state = {
    'walk': False,
    'crouch': False,
    'jump': False,
    'left_click': False,
    'right_click': False
}

# —— Smoothing & thresholds ——
ALPHA                   = 0.4
HORIZONTAL_SENSITIVITY  = 1.0
VERTICAL_SENSITIVITY    = 1.0
LOOK_SCALE              = 35
LOOK_DEADZONE           = 0.05
ARM_RAISE_THRESHOLD     = 0.15  # wrist above shoulder
ARM_DOWN_THRESHOLD      = 0.10  # wrist below shoulder
CROUCH_KNEE_HIP_DIFF    = 0.15  # knee_y - hip_y < diff
KNEE_THRESHOLD          = 0.05  # left_knee above right

# —— COCO keypoint indices ——
IDX_NOSE       = 0
IDX_L_SHOULDER = 5
IDX_R_SHOULDER = 6
IDX_L_ELBOW    = 7
IDX_R_ELBOW    = 8
IDX_L_WRIST    = 9
IDX_R_WRIST    = 10
IDX_L_HIP      = 11
IDX_R_HIP      = 12
IDX_L_KNEE     = 13
IDX_R_KNEE     = 14
IDX_L_ANKLE    = 15
IDX_R_ANKLE    = 16

# —— Skeleton edges ——
SKELETON = [
    (0,1),(0,2), (1,3),(2,4),
    (5,6),(5,7),(7,9), (6,8),(8,10),
    (11,12),(11,13),(13,15),(12,14),(14,16)
]

# —— Window & focus check ——
GetFG = ctypes.windll.user32.GetForegroundWindow
GetTxtW = ctypes.windll.user32.GetWindowTextW
GetTxtL = ctypes.windll.user32.GetWindowTextLengthW
def active_window_title():
    h = GetFG()
    l = GetTxtL(h)
    buf = ctypes.create_unicode_buffer(l+1)
    GetTxtW(h, buf, l+1)
    return buf.value
def in_minecraft():
    return "Minecraft" in active_window_title()

# —— Camera setup ——
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")
cv2.namedWindow("MC Controller", cv2.WINDOW_NORMAL)

prev_smooth = None
center_sh_x = None
center_sh_y = None

while True:
    t0, ret = time.time(), None
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # —— Pose estimation ——
    img = cv2.resize(frame, (INPUT_SZ, INPUT_SZ))
    inp = np.expand_dims(img, 0).astype(np.uint8)
    interpreter.set_tensor(inp_det['index'], inp)
    interpreter.invoke()
    kps = interpreter.get_tensor(out_det['index'])[0,0,:,:]

    pos, conf = kps[:,:2], kps[:,2]
    if prev_smooth is None:
        smooth = old = pos.copy()
    else:
        old    = prev_smooth.copy()
        smooth = ALPHA*pos + (1-ALPHA)*prev_smooth
    prev_smooth = smooth.copy()

    # 1) WALK
    lk_y = smooth[IDX_L_KNEE,0]
    rk_y = smooth[IDX_R_KNEE,0]
    walking = (lk_y - rk_y) > KNEE_THRESHOLD
    if walking != state['walk'] and in_minecraft():
        (safe_press if walking else safe_release)(KEY_W)
    state['walk'] = walking

    # 2) CROUCH (squat)
    hip_y  = (smooth[IDX_L_HIP,0] + smooth[IDX_R_HIP,0]) / 2
    knee_y = (smooth[IDX_L_KNEE,0] + smooth[IDX_R_KNEE,0]) / 2
    crouch = (knee_y - hip_y) < CROUCH_KNEE_HIP_DIFF
    if crouch and not state['crouch'] and in_minecraft():
        safe_press(KEY_CROUCH); safe_release(KEY_CROUCH)
    state['crouch'] = crouch

    # 3) JUMP
    head_y = smooth[IDX_NOSE,0]
    lw_y, rw_y = smooth[IDX_L_WRIST,0], smooth[IDX_R_WRIST,0]
    jump = (lw_y < head_y) and (rw_y < head_y)
    if jump != state['jump'] and in_minecraft():
        (safe_press if jump else safe_release)(KEY_JUMP)
    state['jump'] = jump

    # 4) LOOK 4-directional
    sh_x = (smooth[IDX_L_SHOULDER,1] + smooth[IDX_R_SHOULDER,1]) / 2
    sh_y = (smooth[IDX_L_SHOULDER,0] + smooth[IDX_R_SHOULDER,0]) / 2
    if center_sh_x is None:
        center_sh_x, center_sh_y = sh_x, sh_y
    dx = dy = 0
    if   sh_x < center_sh_x - LOOK_DEADZONE:
        dx = -int((center_sh_x - LOOK_DEADZONE - sh_x)*LOOK_SCALE*HORIZONTAL_SENSITIVITY)
    elif sh_x > center_sh_x + LOOK_DEADZONE:
        dx =  int((sh_x - (center_sh_x + LOOK_DEADZONE))*LOOK_SCALE*HORIZONTAL_SENSITIVITY)
    if   sh_y < center_sh_y - LOOK_DEADZONE:
        dy = -int((center_sh_y - LOOK_DEADZONE - sh_y)*LOOK_SCALE*VERTICAL_SENSITIVITY)
    elif sh_y > center_sh_y + LOOK_DEADZONE:
        dy =  int((sh_y - (center_sh_y + LOOK_DEADZONE))*LOOK_SCALE*VERTICAL_SENSITIVITY)
    if (dx or dy) and in_minecraft():
        safe_mouse(dx, dy)

    # 5) CLICK gestures
    rs_y = smooth[IDX_R_SHOULDER,0]
    rw_y = smooth[IDX_R_WRIST,0]
    arm_down = (rw_y - rs_y) > ARM_DOWN_THRESHOLD
    arm_up   = (rs_y - rw_y) > ARM_RAISE_THRESHOLD
    if arm_down and not state['right_click'] and in_minecraft():
        click_right()
    state['right_click'] = arm_down
    if arm_up and not state['left_click'] and in_minecraft():
        click_left()
    state['left_click'] = arm_up

    # —— Draw skeleton & keypoints ——
    for (i,j) in SKELETON:
        if conf[i]>0.3 and conf[j]>0.3:
            yi, xi = smooth[i]; yj, xj = smooth[j]
            cv2.line(frame, (int(xi*W),int(yi*H)), (int(xj*W),int(yj*H)), (0,255,0), 2)
    for i in range(17):
        if conf[i]>0.3:
            y,x = smooth[i]; cv2.circle(frame,(int(x*W),int(y*H)),4,(0,255,0),-1)

    fps = 1.0/(time.time()-t0)
    cv2.putText(frame,f"FPS: {fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("MC Controller",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
