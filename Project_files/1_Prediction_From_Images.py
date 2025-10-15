import math
import os

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


# use images
def Images(image):
    # # display image with out prediction
    # plt.imshow(
    #     cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # )  # Convert BGR to RGB for proper display
    # plt.axis("off")  # Remove axis
    # plt.show()
    # return

    # display image with prediction
    results = model(image)  # Run the prediction on the image

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Access the bounding box coordinates directly from the tensor
            x1, y1, x2, y2 = box.xyxy[0][:4]  # Get the first 4 elements from the tensor
            label = int(box.cls[0])  # Get the class label as an integer
            confidence = box.conf[0]  # Confidence score

            if coco_names[label] in ["banana", "apple", "orange", "carrot"]:
                # Draw rectangle
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{coco_names[label]} {confidence:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )  # Convert BGR to RGB for proper display
    plt.axis("off")  # Remove axis
    plt.show()


image = cv2.imread("image_data/Mix_Fruits_1.jpg")

Images(image)


# # Read all images from the folder
# images_path = "image_data/apple/"
# images_path = "image_data/banana/"
# images_path = "image_data/carrot/"
# images_path = "image_data/orange/"

# images = os.listdir(images_path)

# for image in images:
#     image_path = os.path.join(images_path, image)
#     image = cv2.imread(image_path)  # Load the image without processing ICC

#     # change height and width
#     width, height, dimention = image.shape
#     new_width = math.floor(width / 2)
#     new_height = math.floor(height / 2)
#     image = cv2.resize(image, (new_width, new_height))

#     Images(image)
