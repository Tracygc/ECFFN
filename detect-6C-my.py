import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs\LLVIP\LLVIP-yolo11-RGBT-midfusion-MCF-PCFVOV-v2-c3k2-Recurive-Bi-AFPN-e300-8-step47\weights/best.pt') # select your model.pt path
    model.predict(source=r"G:\datasets\yolo_LLVIP\yolo_LLVIP_align-im\images\train/010001.jpg",
                  imgsz=640,
                  project='detects/LLVIP',
                  name='11n-ECFFN-RGB-fr/images/train/010001',
                  show=False,
                  save_frames=True,
                  use_simotm="RGBRGB6C",
                  channels=6,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )