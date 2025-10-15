from ultralytics import YOLO
import cv2
import cvzone # cvzone is use for better detection showing
from sort import * # import everything from sort.py

cap = cv2.VideoCapture("video_data/cars.mp4")  # for videos

model = YOLO("yolov9t.pt")

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

mask = cv2.imread("video_data/Video_masks/cars_mask.png")

# traing instence
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [220, 600, 880, 600]
total_count = []

print("Press the 'esc' key to exit the loop.")
while True:
    success, img = cap.read()
    image_region = cv2.bitwise_and(img, mask)
    
        # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
        # img = cvzone.overlayPNG(img, imgGraphics, (0,0))

    result = model(image_region, stream=True)

    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h

            conf = box.conf[0]  # it give a tensor confidence value
            conf = conf.item()  # Convert tensor to decimal number
            conf = round(conf, 2)  # Round a number to two decimal points

            # show class name above the bounding box
            clsN = int(box.cls[0])

            if coco_names[clsN] in ["car", "motorbike", "bus", "truck"] and conf>0.3:
                # cvzone.putTextRect(
                #     img,
                #     f"{coco_names[clsN]}, {conf}",
                #     (max(0, x1), max(35, y1)),
                #     scale = 0.8,
                #     thickness = 1,
                #     offset = 2
                # )
                cvzone.cornerRect(img, bbox, l=10, t=2, rt=1, colorR=(255,0,0))
                
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)

    # create a counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), thickness=2)

    # track the cars and give them some id's
    for results in resultTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        print(f"<< width: {w}, height:{h} >>")
        # print(results)
        cvzone.putTextRect(img, f"{coco_names[clsN]}-{int(id)}, {conf}", (max(0, x1), max(35, y1)), scale = 0.8, thickness = 1, offset = 2)
        
        # find center_x and center_y
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), radius=3, color=(0, 0, 255), thickness=cv2.FILLED)
        # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        # count cars
        if limits[0]<cx<limits[2] and limits[0]-10<cy<limits[3]+10:
            if id not in total_count:
                total_count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0),2)
                
            
    # show count
    cvzone.putTextRect(img, f"Total cars: {len(total_count)}", (10, 50))
    
    # show count when graphic is added
    # cv2.putText(img, str(len(total_count)), org=(250, 100), cv2.FONT_HERSHEY_PLAIN, fontScale=5, (50, 50, 255), thickness=8)
            
    
    cv2.imshow("image", img)  # show the result

    if cv2.waitKey(1) == 27:  # Exit on pressing 'ESC'
        print("Pressed 'ESC'. Exiting...")
        break