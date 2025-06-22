Open the file "ultralytics-yolo11-main" using PyCharm. In the terminal, run the command: `yolo detect train data="yaml file path" model=yolo12n.pt epochs=300 imgsz=640 batch=16 lr0=0.01 workers=8 patience=20 optimizer=SGD`. After the task is completed, the results will be saved in the "runs/detect/train" path. Then, use the validation test set code. Add the file paths as specified in lines 10, 11, and 13, and run it to obtain a "predictions.txt" file.

Here is the code for the validation test set:
”import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO

yaml_path = r"yaml file path"
test_images_dir = r"Test set file path"

predictions_output_dir = r"Test set result save path"
os.makedirs(predictions_output_dir, exist_ok=True)
predictions_file_path = os.path.join(predictions_output_dir, "predictions.txt")

model = YOLO(r"runs/detect/train14/weights/best.pt")  # Use the best model that has been trained

def predict_and_save(test_images_dir, predictions_file_path):
    # Open the predictions.txt file in write mode
    with open(predictions_file_path, 'w') as predictions_file:
        # Iterate through all the images in the test set
        for img_name in os.listdir(test_images_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_images_dir, img_name)
                results = model.predict(img_path, imgsz=640, conf=0.01)

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls)
                        conf = float(box.conf[0])
                        x_center = float(box.xyxyn[0][0])
                        y_center = float(box.xyxyn[0][1])
                        width = float(box.xyxyn[0][2] - box.xyxyn[0][0])
                        height = float(box.xyxyn[0][3] - box.xyxyn[0][1])

                        # Remove the file extension from img_name when writing to the file
                        img_name_without_ext = os.path.splitext(img_name)[0]

                        # Write to the predictions.txt file
                        predictions_file.write(f"{img_name_without_ext} {cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

predict_and_save(test_images_dir, predictions_file_path)
