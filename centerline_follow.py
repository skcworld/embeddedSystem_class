import torch
import numpy as np
from PIL import Image
from jetcam.csi_camera import CSICamera
from cnn.center_dataset import TEST_TRANSFORMS
from torchvision.models import alexnet
import time

# ===== 모델 정의 및 로드 =====
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

