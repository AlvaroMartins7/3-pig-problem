"""Real-time video inference test for a trained YOLO pig detection model.

Runs object detection on a video file frame-by-frame in streaming mode
and displays the results in a resizable OpenCV window. Press 'q' to quit.
"""

from ultralytics import YOLO
import cv2

# Load the trained model weights
#model = YOLO("experimentos/no_noise/detect/train/weights/best.pt")
model = YOLO("experimentos/noise/detect/train/weights/best.pt")

# Run inference on the video in streaming mode (processes one frame at a time)
results = model("12-Ceiling_Cam.mp4", stream=True)

# Display scale for the output window
resize_scale = 0.5

# Process and display each frame with detection overlays
for result in results:
    frame = result.plot()  # Draw detection boxes and labels on the frame

    # Resize frame to a fixed display resolution
    frame_resized = cv2.resize(frame, (1280, 720))

    cv2.imshow("Detections", frame_resized)

    # Press 'q' to exit the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
