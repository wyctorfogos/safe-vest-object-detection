from ultralytics import YOLO

# Path to the input video file
video_path = 'dataset/videos/Importance of Wearing Safety Helmets at Work.We Are Navigators-connecting seafarers.mp4'

# Load the YOLO model
model = YOLO('/home/wytcor/PROJECTs/SafeVest/runs/detect/train6/weights/last.pt').to('cuda')

# Run the model on the video with streaming enabled
results = model(source=video_path, stream=True)

# Path to save the inferred video
output_path = 'dataset/videos/inferred_video.mp4'

# Open a VideoWriter to save the video
import cv2

# Get video properties
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Initialize the video writer
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process the video frame by frame and save the results
for r in results:
    # r.plot() returns an image with the detections overlaid
    result_frame = r.plot()
    
    # Write the frame to the output video
    out.write(result_frame)
    
# Release the VideoWriter
out.release()

print(f"Inferred video saved to {output_path}")
