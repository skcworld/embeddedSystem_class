import torch
import numpy as np
from PIL import Image
import time
import threading

from jetcam.csi_camera import CSICamera
from cnn.center_dataset import TEST_TRANSFORMS
from torchvision.models import alexnet
from base_ctrl import BaseController  # BaseController 클래스는 /dev/ttyUSB0에 연결

# ========== 제어 파라미터 ==========
MAX_STEER = 4.0
MAX_SPEED = 0.5
Kp = 0.025  # 비례 제어 게인

# ========== 장치 초기화 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model():
    return alexnet(num_classes=2, dropout=0.0)

model = get_model()
model.load_state_dict(torch.load('/home/ircv15/em7/perception/road_following_model2.pth'))
model.to(device)
model.eval()

def preprocess(image: np.ndarray):
    image_pil = Image.fromarray(image)
    tensor = TEST_TRANSFORMS(image_pil).to(device)
    return tensor.unsqueeze(0)

base = BaseController('/dev/ttyUSB0', 115200)

def send_control_async(L, R):
    def worker():
        base.base_json_ctrl({"T": 1, "L": L, "R": R})
        print('help')
    threading.Thread(target=worker).start()

def clip(val, max_val):
    return max(min(val, max_val), -max_val)

def update_vehicle_motion(steering, speed):
    steer_val = -clip(steering, MAX_STEER)
    speed_val = clip(speed, MAX_SPEED)

    base_speed = abs(speed_val)
    left_ratio = max(1.0- steer_val, 0)
    right_ratio = max(1.0+steer_val, 0)
        
    L = clip(base_speed * left_ratio, MAX_SPEED)
    R = clip(base_speed * right_ratio, MAX_SPEED)

    if speed_val > 0:  # 전진 (하드웨어 기준 반대)
         L, R = -L, -R

    send_control_async(-L, -R)
    print(f"[UGV] Steering={steer_val:.2f}, Speed={speed_val:.2f} → L={L:.2f}, R={R:.2f}", flush=True)

# ========== 카메라 연결 ==========
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)
time.sleep(2.0)

# 초기 프레임 버퍼 제거
for _ in range(10):
    _ = camera.read()
    time.sleep(0.05)

# ========== 메인 루프 시작 ==========
try:
    print("🚗 실시간 중앙선 추종 자율주행 시작 (종료: Ctrl+C)", flush=True)
    fixed_speed = 0.2  # 전진 (하드웨어 기준)

    while True:
        frame = camera.read()
        height, width = frame.shape[:2]
        x_center = (width / 2)-100

        with torch.no_grad():
            input_tensor = preprocess(frame)
            output = model(input_tensor).cpu().numpy()[0]

        x = (output[0] / 2 + 0.5) * width
        y = (output[1] / 2 + 0.5) * height
        if x>640:
            x_center=640    
        elif x<440:
            x_center=440
        else:
            x_center=540            
        x_error = x - x_center
        steering = clip(Kp * x_error, MAX_STEER)

        update_vehicle_motion(steering, fixed_speed)
        # base.base_json_ctrl({"T": 1, "L": 0.5, "R": 0.5})
        time.sleep(0.03333)

        print(f"📍 Pred: ({int(x)}, {int(y)}), x_err={x_error:.1f}, steer={steering:.3f}", flush=True)
        # time.sleep(1/30)  # 20Hz 제어 주기

except KeyboardInterrupt:
    print("\n🛑 사용자 종료. 모터 정지.", flush=True)
    base.base_json_ctrl({"T": 1, "L": 0.0, "R": 0.0})
