# First steps
conda create -n safeVestEnv

conda activate safeVestEnv

pip install -U ultralytics


# Download the dataset
Go to the web page of Universe Roboflow `https://universe.roboflow.com/priyam-sheth/safety_equipments-3brcs/dataset/3` and put it inside the folder dataset

# Run script
After the first steps,  on the main path "/", run the command:
`python3 src/scripts/train_models.py`


# Inference
To test the trained models, download it and run the script `python3 src/scripts/infer.py`

Example:
video: https://youtu.be/5-tAZy62vCY



![Wearing object detection](https://github.com/wyctorfogos/safe-vest-object-detection/blob/master/val_batch1_pred.jpg)
