# First steps
conda create -n safeVestEnv

conda activate safeVestEnv

pip install -U ultralytics


# Download the dataset
Go to the web page of Universe Roboflow `https://universe.roboflow.com/priyam-sheth/safety_equipments-3brcs/dataset/3` and put it inside the folder dataset

# Run script
After the first steps,  on the main path "/", run the command:
`python3 src/scripts/train_models.py`