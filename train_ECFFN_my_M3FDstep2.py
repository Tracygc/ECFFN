import warnings

import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    '''
    利用Dahang Wan的YOLOV11-RGBT论文提到的MCF进行训练  

        Step 1. 加载预训练权重,或者利用其他方式,得到一个 单模态或者单光谱效果比较好的检测模型，当做主分支冻结部分的权重,用于第3步的网络权重转换（建议红外和可将光都训练一个，用于确定主分支）
        Step 2. 设置   epochs=1,  fraction=0.01, # 仅用  训练一个随机初始化的网络权重,用于第3步的网络权重转换
        Step 3. 将第一步得到的模型权重加载到第二步的网络结构中，并将 ZeroConv2d 部分的权重清零，得到 yolo11n-RGBT-midfussion-MCF.pt
        Step  4. 将第三步得到的模型直接用于训练，不要加载yaml，直接加载   yolo11n-RGBT-midfussion-MCF.pt 文件 进行训练


    We will later explore simpler training methods. For now, we will conduct MCF training using the following approach. 
        Step 1. Load the pre-trained weights, or use other methods to obtain a detection model with better single-modal or single-spectrum performance, and use the weights of the frozen part of the main branch as the weights for the network weight conversion in the third step (it is recommended to train both infrared and visible light separately for this purpose, to determine the main branch).
        Step 2. Set epochs = 1, fraction = 0.01. # Only use to train a randomly initialized network weight for the network weight conversion in the third step.
        Step 3. Load the model weights obtained in the first step into the network structure of the second step, and clear the weights of the ZeroConv2d part, obtaining yolo11n-RGBT-midfussion-MCF.pt.
        Step 4. Use the model obtained in the third step directly for training, do not load the yaml file, and directly load the yolo11n-RGBT-midfussion-MCF.pt file for training.


    '''

    # Step 2
    model = YOLO('ultralytics/my_cfg/models/11-RGBT-im/yolo11n-ECFFN.yaml')
    model.train(data='ultralytics/my_cfg/data/M3FD_RGBT.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=16,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD',
                # resume='', # last.pt path
                # amp=False, # close amp
                fraction=0.01,
                # 仅用 1% 的数据训练, 快速得到一个模型权重模板    Train with only 1% of the data. Quickly obtain a model weight template
                use_simotm="RGBRGB6C",
                channels=6,
                project='runs/M3FD',
                name='M3FD-yolo11n-ECFFN-step2',
                # name='M3FD-yolo11n-RGBT-midfusion-MCF-e300-16-',
                )
    del model
    torch.cuda.empty_cache()

