import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'F:\yolo+ir+rgb\YOLOv11_RGBT_master0619\YOLOv11_RGBT0\pre_weights\LLVIP/LLVIP-yolo11n-e300-16-pretrained-.pt',) # select your model.pt path
    # model = YOLO(r'F:\yolo+ir+rgb\YOLOv11_RGBT_master0619\YOLOv11_RGBT0\pre_weights\LLVIP/LLVIP_IF-yolo11n-e300-16-pretrained-.pt', )
    model.predict(source=r"G:\datasets\yolo_LLVIP\yolo_LLVIP_align-im\images\train/010001.jpg",
                  imgsz=640,
                  project='detects/LLVIP',
                  name='11n-pretrained/images/train/010001',
                  show=False,
                  save_frames=True,
                  use_simotm="RGB",
                  channels=3,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )