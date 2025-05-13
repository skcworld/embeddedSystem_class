from base_ctrl import BaseController
import time
import threading
import torch
import numpy as np
from PIL import Image
from jetcam.csi_camera import CSICamera
from cnn.center_dataset import TEST_TRANSFORMS
from torchvision.models import alexnet

def get_model():
    return alexnet(num_classes=2, dropout=0.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model()
model.load_state_dict(torch.load('/home/ircv15/em7/perception/road_following_model.pth'))
model.to(device)
model.eval()

# ===== 전처리 함수 =====
def preprocess(image: np.ndarray):
    image_pil = Image.fromarray(image)  # BGR → PIL (RGB 자동 처리됨)
    tensor = TEST_TRANSFORMS(image_pil).to(device)
    return tensor.unsqueeze(0)

# ===== CSI 카메라 시작 =====
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

time.sleep(2.0)  # 카메라 워밍업

print("🚗 실시간 중앙선 좌표 추정 시작! (종료: Ctrl+C)")

try:
    while True:
        frame = camera.read()
        height, width = frame.shape[:2]

        with torch.no_grad():
            input_tensor = preprocess(frame)
            output = model(input_tensor).cpu().numpy()[0]

        # 좌표 복원
        x = (output[0] / 2 + 0.5) * width
        y = (output[1] / 2 + 0.5) * height

        print(f"📍 Predicted center: ({int(x)}, {int(y)})")
        time.sleep(0.05)  # 20Hz 속도 제한 (optional)

except KeyboardInterrupt:
    print("\n🛑 중단되었습니다. 카메라 정리 중...")
    camera.cap.release()




# === Constants ===
MAX_STEER = 0.8
MAX_SPEED = 0.5
STEP_STEER = 0.2
STEP_SPEED = 0.05

base = BaseController('/dev/ttyUSB0', 115200)

steering = 0.0
speed = -0.5

update_interval = 0.05  # 0.05초로 대기 시간 짧게 설정

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
    threading.Thread(target=worker).start()

def clip(val, max_val):
    return max(min(val, max_val), -max_val)

def update_vehicle_motion(steering, speed):
    steer_val = clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_STEER)

    base_speed = abs(speed_val)

    left_ratio = 1.0 - steer_val
    right_ratio = 1.0 + steer_val

    if left_ratio < 0:
        left_ratio = 0
    elif right_ratio < 0:
        right_ratio = 0

    L = base_speed * left_ratio
    R = base_speed * right_ratio

    L = clip(L, MAX_SPEED)
    R = clip(R, MAX_SPEED)

    if speed < 0:
        L, R = -L, -R

    send_control_async(-L, -R)
    print(f"[UGV] Speed: {speed_val:.2f}, Steering: {steer_val:.2f} → L: {L:.2f}, R: {R:.2f}")

# === 자동 제어 사이클 ===
speed_sequence = [STEP_SPEED, -STEP_SPEED, -STEP_SPEED, STEP_SPEED]
steer_sequence = [-STEP_STEER, STEP_STEER, -STEP_STEER, STEP_STEER]
seq_len = len(speed_sequence)
idx = 0

try:
    for cycle in range(2000):  # 100회만 반복
        speed += speed_sequence[idx]
        steering += steer_sequence[idx]

        update_vehicle_motion(steering, speed)

        idx = (idx + 1) % seq_len
        time.sleep(update_interval)  # 0.05초만 대기

    print("\n[UGV] 100 사이클 완료. 모터 정지.")
    base.base_json_ctrl({"T": 1, "L": 0.5, "R": 0.5})

except KeyboardInterrupt:
    print("\n[UGV] 강제 종료됨. 모터 정지.")
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})


    
