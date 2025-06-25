This project utilizes the YOLO model to conduct training and prediction for object detection tasks. Here are the detailed steps:

 Environment Preparation

Before starting the training, ensure the following dependencies are installed:
- Python 3.12.11
- torch 2.7.1+cu118
- torchvision 0.22.1+cu118
- numpy 2.1.2
- opencv-python 4.11.0.86
- pip 25.1
- timm 1.0.15
- dill 0.4.0
- psutil 7.0.0
- PyCharm (or other Python development environments)
- ultralytics 8.3.152

I integrated multiple files into a new dataset folder. It includes a training set (train), a validation set (val), and a unified tree class file (class_all, uploaded to GitHub). Under both the training and validation directories, I created images and labels folders. The corresponding images and labels from files like 0_RGB_FullyLabeled, 12_RGB_ObjDet_640_fL, 34_RGB_ObjDet_640_pL, and 34_RGB_ObjDet_640_pL_b were placed into these folders. This organized dataset facilitates subsequent model training and validation.


Training the Model

1. Open PyCharm and import the `ultralytics-main` project file.

2. Edit the `train.py` file to specify the following parameters:
   - Model path: `model = YOLO('ultralytics/cfg/models/12/yolo12m.yaml')`.
   - Dataset path: In the `model.train()` function, replace `data='YAML file path'` with the actual path to the YAML file (the 12rgb-objdet.yaml), which should define the paths for the training, validation, and test sets.

3. Configure the training parameters:
   - `imgsz`: Set the image size to 640.
   - `epochs`: Set the number of training epochs to 300.
   - `batch`: Set the batch size to 32.
   - `close_mosaic`: Set to 0 to disable mosaic data augmentation.
   - `workers`: Set the number of worker processes to 8.
   - `optimizer`: Choose SGD (Stochastic Gradient Descent) as the optimizer.
   - `project`: Specify the project save path as `runs/train`.
   - `name`: Specify the experiment name as `exp`.

4. Run the `train.py` file to start the training process. The training results, including logs and model weight files, will be saved in the `runs/train/exp` directory (the exact path can be checked in the run box).

Generating Predictions

1. After training is complete, load the best model path in the predictions code:
   `Model = YOLO('The best model path')`

2. Use the trained model for prediction by specifying the path to the images to be predicted in `test_images_dir = 'The path to the best set image folder'`.

3. The validation test set code can be obtained from GitHub or Dropbox. Edit the validation test set code by adding the corresponding file paths on lines 6, 8, and 10, then run the code.

4. After running the validation test set code, a `predictions.txt` file will be generated, containing the model's prediction results for the test set.

 File Structure Description

- `ultralytics-main`: The main project folder.
- `model weights`: The folder for storing model weight files.
- `ultralytics-main/runs/detect/train/results.csv`: The file storing model training accuracy.
- `validation test set code`: The validation test set code file.
- `predictions.txt`: The file storing prediction results.
- `all_objdet`: The folder containing the training set, validation set, and test set.

 Notes

- If you need to load pre-trained weights, uncomment `model.load('yolo12n.pt')` and replace `yolo12n.pt` with the actual pre-trained weight file path.
- Ensure the YAML file correctly specifies the paths for the training, validation, and test sets to avoid data loading errors.
- If you encounter memory insufficiency issues during the process, try reducing the value of the `batch` parameter.

By following the above steps, you can successfully train the YOLO model and generate prediction results for object detection tasks. If you encounter any problems during the process, please refer to the relevant documentation or contact the project provider for support.
