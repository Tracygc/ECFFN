import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs\LLVIP\LLVIP-yolo11n-ECFFN-e300-8-step48\weights/best.pt')
    model.val(data='ultralytics/my_cfg/data/LLVIP_RGBT.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # use_simotm="RGBT",
              # channels=4,
              # use_simotm="RGB",
              # channels=3,
              use_simotm="RGBRGB6C",
              channels=6,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='vals/LLVIP_RGBT',
              name='LLVIP-yolo11n-ECFFN-e300-8-step48-val.pt', # 实际batchsize是8(true8)
              )