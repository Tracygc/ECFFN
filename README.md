# ECFFN
ECFFN: Efficient Cross-modality Feature Fusion Network for Multispectral Fusion Detection


### Dataset Structure
```
dataset/
├── images/
│   ├── visible/
│   │   ├── train/  # Store training visible light images
│   │   └── val/    # Store validation visible light images
│   └── infrared/
│       ├── train/  # Store training infrared images
│       └── val/    # Store validation infrared images
└── labels/
    ├── visible/
    │   ├── train/  # Store training visible light image labels
    │   └── val/    # Store validation visible light image labels
    └── infrared/
        ├── train/  # Store training infrared image labels
        └── val/    # Store validation infrared image labels
---------------------------------------------------------------------
# FLIR_aligned.yaml  (for aligned iimages or RGB images)

train: G:/datasets/FLIR-align-3class/FLIR-align-3class/images/visible/train # 128 images
val: G:/datasets/FLIR-align-3class/FLIR-align-3class/images/visible/test # 128 images


# number of classes
nc: 3

# class names
names: ["person", "car", "bicycle"]
-----------------------------------------------------------------------
# FLIR_aligned_IF.yaml  (for infrared images)

train: G:/datasets/FLIR-align-3class/FLIR-align-3class/images/infrared/train # 128 images
val: G:/datasets/FLIR-align-3class/FLIR-align-3class/images/infrared/test # 128 images


# number of classes
nc: 3

# class names
names: ["person", "car", "bicycle"]
-----------------------------------------------------------------------

```
### Install Dependencies
Install the ultralytics package, including all requirements, in a Python>=3.8 environment with PyTorch>=1.8.
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install ultralytics
```
### Python
MCF train
Step 1. Load the pre-trained weights, or use other methods to obtain a detection model with better single-modal or single-spectrum performance, and use the weights of the frozen part of the main branch as the weights for the network weight conversion in the third step (it is recommended to train both infrared and visible light separately for this purpose, to determine the main branch).
Step 2. Set epochs = 1, fraction = 0.01. # Only use to train a randomly initialized network weight for the network weight conversion in the third step.
Step 3. Load the model weights obtained in the first step into the network structure of the second step, and clear the weights of the ZeroConv2d part, obtaining yolo11n-RGBT-midfussion-MCF.pt.
Step 4. Use the model obtained in the third step directly for training, do not load the yaml file, and directly load the yolo11n-RGBT-midfussion-MCF.pt file for training.

Take the M3FD dataset as an example, and proceed with the following steps for training and testing in sequence.        
```
python train_ECFFN_my_M3FDstep1.py
python train_ECFFN_my_M3FDstep2.py.py
python train_ECFFN_my_M3FDstep3.py.py
python train_ECFFN_my_M3FDstep4.py.py
python val_M3FD.py
```

### Reference Links
```
https://docs.ultralytics.com/
https://github.com/wandahangFY/YOLOv11-RGBT
```
