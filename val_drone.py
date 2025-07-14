import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO('runs/DroneVehicle/5.DroneVehicle-yolo12l-IR\weights/best.pt')
    # model.val(data='ultralytics\my_cfg\data\DroneVehicle_aligned_IF_RGBT.yaml',
    #           split='val',
    #           imgsz=640,
    #           batch=8,
    #           # use_simotm="RGBT",
    #           # channels=4,
    #           use_simotm="RGB",
    #           channels=3,
    #           # use_simotm="RGBRGB6C",
    #           # channels=6,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           project='vals/DroneVehicle',
    #           name='DroneVehicle-yolo12l-IR-val', # 实际batchsize是8(true8)
    #           )

    model = YOLO('runs\DroneVehicle\8.DroneVehicle-yolo11n-RGB\weights/best.pt')
    model.val(data='ultralytics\my_cfg\data\DroneVehicle_aligned_RGBT.yaml',
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
              project='vals/DroneVehicle',
              name='DroneVehicle-yolo11n-RGB-e300-8-val', # 实际batchsize是8(true8)
              )