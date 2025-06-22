Open the "ultralytics-yolo11-main" file in PyCharm. In the terminal, run the command: `yolo detect train data="yaml file path" model=yolo12n.pt epochs=300 imgsz=640 batch=16 lr0=0.01 workers=8 patience=20 optimizer=SGD`. After the task finishes, the results are saved in the "runs/detect/train" path. Then, use the validation test set code, add the file paths as per lines 7, 8, and 10, and run it to generate a "predictions.txt" file.

1. The "ultralytics-yolo11-main" file has been uploaded to Dropbox.
2. The model weights file can be found in the Dropbox file or the provided GitHub link.
3. The model accuracy file can be found in "Model accuracy.txt" or "model weights\results.csv".
4. After opening "ultralytics-yolo11-main", the results in the "runs/detect/train" path can be viewed. If not found, check the results path in the run log after executing the code.
5. The validation test set code is in the submitted "Validate test set code" file or the GitHub "Validate test set code" file.
6. The "predictions.txt" file can be found in the Dropbox file or the provided GitHub link.
