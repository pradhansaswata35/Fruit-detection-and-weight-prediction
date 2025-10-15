import math

from ultralytics import YOLO
import cv2
import cvzone
from matplotlib import pyplot as plt

model = YOLO("yolov9t.pt")  # Load the YOLO model


coco_names = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "apple",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

import cv2
import numpy as np

def detect_apple_color(image, x1, y1, x2, y2):
    """
    Determines if the fruit in the bounding box is Red, Green, or Red_and_Green.
    Uses HSV masking and average BGR analysis.
    """
    # Crop the region of interest (ROI)
    roi = image[int(y1):int(y2), int(x1):int(x2)]

    # Convert to HSV for red masking
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Create masks for red color (both ends of the hue spectrum)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)

    # Combine red masks
    red_mask = mask1 + mask2

    # Find the largest red area
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop red region inside ROI
        red_region = roi[y:y+h, x:x+w]

        # Calculate average BGR color of the red region
        avg_color = cv2.mean(red_region)[:3]
    else:
        # No red area found, use entire ROI
        avg_color = cv2.mean(roi)[:3]

    b, g, r = avg_color

    # Simple decision logic
    if r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    else:
        return "red_and_green"


def apple_weight_prediction(height, width, color):
    import scaler_utils_for_apple as su_apple
    min_vals = su_apple.min_vals
    scale_vals = su_apple.scale_vals
    
    scaled_height = (height - min_vals[0]) / scale_vals[0]
    scaled_width = (width - min_vals[1]) / scale_vals[1]
    
    if color == "Red":
        red, green = 1, 0
    elif color == "Green":
        red, green = 0, 1
    else:
        red, green = 0, 0

    apple_info = [scaled_height, scaled_width, red, green]
    
    import joblib
    model = joblib.load("Trained_model/Weights_Prediction_Apple")
    
    result = model.predict([apple_info])
    return result
    
    
import cv2
import numpy as np

def detect_banana_color(image, x1, y1, x2, y2):
    """
    Detects the color of a banana in the given bounding box.
    Returns: 'yellow', 'green', or 'yellow_and_green'
    """
    roi = image[int(y1):int(y2), int(x1):int(x2)]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    # Green color range in HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)

    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if yellow_pixels > green_pixels and yellow_pixels > 200:
        return "yellow"
    elif green_pixels > yellow_pixels and green_pixels > 200:
        return "green"
    else:
        return "yellow_and_green"

def banana_weight_prediction(height, width, color):
    import scaler_utils_for_banana as su_banana
    min_vals = su_banana.min_vals
    scale_vals = su_banana.scale_vals
    
    scaled_height = (height - min_vals[0]) / scale_vals[0]
    scaled_width = (width - min_vals[1]) / scale_vals[1]
    
    if color == "yellow":
        red, green = 1, 0
    elif color == "Green":
        red, green = 0, 1
    else:
        red, green = 0, 0

    banana_info = [scaled_height, scaled_width, red, green]
    
    import joblib
    model = joblib.load("Trained_model/Weights_Prediction_Banana")
    
    result = model.predict([banana_info])
    return result
    
import cv2
import numpy as np

def detect_carrot_color(image, x1, y1, x2, y2):
    """
    Detects the color of a carrot in the given bounding box.
    Returns: 'orange', 'purple', or 'yellow'
    """
    roi = image[int(y1):int(y2), int(x1):int(x2)]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Orange color range
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv_roi, lower_orange, upper_orange)

    # Purple color range (HSV for dark magenta)
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([160, 255, 255])
    purple_mask = cv2.inRange(hsv_roi, lower_purple, upper_purple)

    # Yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

    orange_pixels = cv2.countNonZero(orange_mask)
    purple_pixels = cv2.countNonZero(purple_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # Choose the dominant color
    if orange_pixels > purple_pixels and orange_pixels > yellow_pixels and orange_pixels > 150:
        return "orange"
    elif purple_pixels > orange_pixels and purple_pixels > yellow_pixels and purple_pixels > 150:
        return "purple"
    elif yellow_pixels > orange_pixels and yellow_pixels > purple_pixels and yellow_pixels > 150:
        return "yellow"
    else:
        return "undetermined"

def carrot_weight_prediction(height, width, color):
    import scaler_utils_for_carrot as su_carrot
    min_vals = su_carrot.min_vals
    scale_vals = su_carrot.scale_vals
    
    scaled_height = (height - min_vals[0]) / scale_vals[0]
    scaled_width = (width - min_vals[1]) / scale_vals[1]
    
    if color == "orange":
        red, green = 1, 0
    elif color == "purple":
        red, green = 0, 1
    else:
        red, green = 0, 0

    carrot_info = [scaled_height, scaled_width, red, green]
    
    import joblib
    model = joblib.load("Trained_model/Weights_Prediction_Carrot")
    
    result = model.predict([carrot_info])
    return result
    
import cv2
import numpy as np

def detect_orange_color(image, x1, y1, x2, y2):
    """
    Detects the color of an orange in the given bounding box.
    Returns: 'orange', 'green', or 'orange_and_green'
    """
    roi = image[int(y1):int(y2), int(x1):int(x2)]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Orange color range
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv_roi, lower_orange, upper_orange)

    # Green color range
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)

    # Count non-zero pixels in masks
    orange_pixels = cv2.countNonZero(orange_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if orange_pixels > green_pixels and orange_pixels > 200:
        return "orange"
    elif green_pixels > orange_pixels and green_pixels > 200:
        return "green"
    else:
        return "orange_and_green"

def orange_weight_prediction(height, width, color):
    import scaler_utils_for_orange as su_orange
    min_vals = su_orange.min_vals
    scale_vals = su_orange.scale_vals
    
    scaled_height = (height - min_vals[0]) / scale_vals[0]
    scaled_width = (width - min_vals[1]) / scale_vals[1]
    
    if color == "orange":
        red, green = 1, 0
    elif color == "green":
        red, green = 0, 1
    else:
        red, green = 0, 0

    orange_info = [scaled_height, scaled_width, red, green]
    
    import joblib
    model = joblib.load("Trained_model/Weights_Prediction_Orange")
    
    result = model.predict([orange_info])
    return result
    
# cap = cv2.VideoCapture(0)  # create a webcam object
# cap.set(3, 640)  # set width
# cap.set(4, 480)  # set height

# cap = cv2.VideoCapture("video_data/cars.mp4")  # use prerecodel video

cap = cv2.VideoCapture("http://100.64.151.189:8080/video")

def Webcam_Video():
    captured_images = []  # list to store captured frames
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        cv2.imshow("image", img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            print("ESC pressed — Exiting...")
            break

        elif key == 32:  # Spacebar key
            # Clone the current frame and store
            captured_images.append(img.copy())
            print(f"📸 Captured image {len(captured_images)}")
    
    cap.release()
    cv2.destroyAllWindows()

    # print(captured_images)
    return captured_images  # returns the list of captured image objects

def Captured_Image_Prediction():
    images = Webcam_Video()
    
    detected_classes = []
    detected_weights = []
    
    sln = 0
    predictions = []
    total_weight = 0

    for image in images:
        image = cv2.resize(image, (640, 640))

        results = model(image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0][:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = coco_names[label]

                if class_name in ["banana", "apple", "orange", "carrot"]:
                    detected_classes.append(class_name)
                    height = y2 - y1
                    width = x2 - x1

                    # Determine color and predict weight
                    if class_name == "apple":
                        color = detect_apple_color(image, x1, y1, x2, y2)
                        weight = apple_weight_prediction(height, width, color)
                    elif class_name == "banana":
                        color = detect_banana_color(image, x1, y1, x2, y2)
                        weight = banana_weight_prediction(height, width, color)
                    elif class_name == "carrot":
                        color = detect_carrot_color(image, x1, y1, x2, y2)
                        weight = carrot_weight_prediction(height, width, color)
                    elif class_name == "orange":
                        color = detect_orange_color(image, x1, y1, x2, y2)
                        weight = orange_weight_prediction(height, width, color)

                sln += 1
                total_weight += weight

                predictions.append({
                    "class": str(class_name.capitalize()),
                    "height": int(height),
                    "width": int(width),
                    "color": str(color),
                    "weight": int(weight)
                })

    return {
        "status": "success",
        "data": predictions,
        "total_weight": int(total_weight)
    }


# Captured_Image_Prediction()






























