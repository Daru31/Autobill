import cv2
from ultralytics import YOLO

# model = YOLO('yolov8s.pt')

model = YOLO('./best.pt')
results = model('./images/3073265_1.jpg') # conf=0.2, iou ..

plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows() 