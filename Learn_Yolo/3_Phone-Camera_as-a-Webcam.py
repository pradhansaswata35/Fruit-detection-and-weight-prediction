import cv2

url = "http://192.168.247.242:8080/video"  # Example URL (replace with your actual URL)

# Create a VideoCapture object with the IP Webcam URL
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Unable to connect to IP Webcam. Please check the URL.")
    exit()

print("Press 'ESC' to exit the video stream.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    # Display the video stream
    cv2.imshow("Mobile Camera as Webcam", frame)

    # Break the loop if 'ESC' is pressed
    if cv2.waitKey(1) == 27:  # ASCII code for 'ESC' key is 27
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
