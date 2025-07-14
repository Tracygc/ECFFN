import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs\M3FD_im\M3FD_im-yolo11n-RGBT-midfusion-e300-8-step4\weights/best.pt')
    model.val(data=r'ultralytics/my_cfg/data/M3FD_RGBT.yaml',
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
              project='vals/M3FD_im',
              name='M3FD_im-yolo11n-RGBT-midfusion-e300-8-step4-val.pt', # 实际batchsize是8(true8)
              )