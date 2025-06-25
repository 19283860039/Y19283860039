import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
torch.cuda.empty_cache()

if __name__ == '__main__':
    torch.cuda.empty_cache()

    model = YOLO('ultralytics/cfg/models/12/yolo12m.yaml')
    # model.load('yolo12n.pt') # loading pretrain weights
    model.train(data='YAML file path',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0, 
                workers=8, 
                optimizer='SGD', 
                project='runs/train',
                name='exp',
                )
