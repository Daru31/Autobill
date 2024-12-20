import cv2
from ultralytics import YOLO 
import threading
import time

# Load the YOLO model
model = YOLO('best.pt') 
camera_index = 1  # Use 0 for the default webcam, change to 1 or higher for other connected cameras
file_index = 1 

def run_tracker_in_thread(camera_index, model, file_index):
    # Open the webcam
    video = cv2.VideoCapture(camera_index)  # 0 for default webcam, 1 for second webcam, etc.

    if not video.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = video.read()  # Read the webcam frame

        # Exit the loop if no frame is captured (camera might have been disconnected)
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Track objects in frames using the YOLO model
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()

        # Show the results in a window
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        # Check for user exit key (press 'q' to exit)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    video.release()
    cv2.destroyAllWindows()

# Create a thread to run the tracker
tracker_thread1 = threading.Thread(target=run_tracker_in_thread,
                                   args=(camera_index, model, file_index),
                                   daemon=True)

tracker_thread1.start()

# Keep the main thread alive while the tracker thread is running
while tracker_thread1.is_alive():
    time.sleep(1)


"""
This function is designed to run a video file or webcam stream
concurrently with the YOLOv8 model, utilizing threading.

- filename: The path to the video file or the webcam/external
camera source.
- model: The file path to the YOLOv8 model.
- file_index: An argument to specify the count of the
file being processed.
"""