from ultralytics import YOLO
import cv2

# create a object of yolo model(yolov9m). if it is not downloaded it will download it automatically.
model = YOLO("yolov9t.pt")

# 'image_data/fruits.jpg' is imput image source. show=True is used to show the result/output, but by default it open and close very quickly
result = model("image_data/fruits.jpg", show=True)

# it help to prevent from quick open and close problem. '0' means unless the user give any input don't do any thing, in our case it is manual output windows closing.
cv2.waitKey(0)
