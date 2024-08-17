from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = YOLO('yolov8m.pt')


# Train the model with the specified dataset and number of epochs
results = model.train(data="/home/wytcor/PROJECTs/SafeVest/dataset/Safety_equipments.v3i.yolov8/data.yaml", epochs=10, batch=8)

# Validate the model (you can specify a validation dataset if needed)
val_results = model.val()

# Export the trained model to ONNX format
model.export(format="onnx")

# Use the model to make predictions on a new image
results = model("https://ultralytics.com/images/bus.jpg")

# You can print or visualize the results
print(results)
