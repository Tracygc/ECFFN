import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs\FLIR_im/FLIRim-yolo11l-RGB2\weights/best.pt')
    model.val(data=r'ultralytics/my_cfg/data/FLIR_aligned_RGBT.yaml',
    # model.val(data=r'F:/yolo+ir+rgb/YOLOv11_RGBT_master0619/YOLOv11_RGBT/ultralytics/my_cfg/data/FLIR_aligned_IF_RGBT.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # use_simotm="RGBT",
              # channels=4,
              use_simotm="RGB",
              channels=3,
              # use_simotm="RGBRGB6C",
              # channels=6,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='vals/FLIR_im',
              name='FLIRim-yolo11l-RGB2-val',
              )
