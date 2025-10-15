from ultralytics import YOLO
import cv2
import cvzone  # cvzone is use for better detection showing
import math

url = "http://192.168.247.242:8080/video"  # Example URL (replace with your actual URL)

# Create a VideoCapture object with the IP Webcam URL
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Unable to connect to IP Webcam. Please check the URL.")
    exit()

print("Press 'ESC' to exit the video stream.")

model = YOLO("yolov9m.pt")

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
    "elephant",
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

print("Press the 'esc' key to exit the loop.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break
    
    success, img = cap.read()
    result = model(img, stream=True)

    for r in result:
        boxes = r.boxes
        for box in boxes:

            # show bounding box using cv2
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # print(f"<< {x1}, {y1}, {x2}, {y2} >>")

            # show bounding box using cvzone
            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h
            cvzone.cornerRect(img, bbox)
            # print(f"<< {x1}, {y1}, {w}, {h} >>")

            conf = box.conf[0]  # it give a tensor confidence value
            conf = conf.item()  # Convert tensor to decimal number
            conf = round(conf, 2)  # Round a number to two decimal points

            # show confidence value above the bounding box
            # cvzone.putTextRect(img, f"{conf}", (x1, y1 - 20))

            # to solve out_of_the_screen confidence value problem
            # cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(30, y1 - 20)))

            # show class name above the bounding box
            clsN = int(box.cls[0])
            # print(f"class_No:{clsN}, class_Name:{coco_names[clsN]}")
            cvzone.putTextRect(
                img,
                f"{coco_names[clsN]}, {conf}",
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1,
            )

    cv2.imshow("image", img)  # show the result

    if cv2.waitKey(1) == 27:  # Exit on pressing 'ESC'
        print("Pressed 'ESC'. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()