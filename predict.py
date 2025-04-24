import cv2
from ultralytics import YOLO
image_path = "6.jpeg"
model = YOLO("runs/classify/train/weights/best.pt")
results = model.predict(image_path)
result = results[0]  # Access the first (and possibly only) result

probs = result.probs

predicted_class_index = probs.top1  # This is the index of the predicted class

# Get the confidence of the top prediction
confidence = probs.top1conf  # Confidence of the predicted class

# Get the class name using the index
predicted_class = result.names[predicted_class_index]  # Get predicted class name

# Load the image
image = cv2.imread(image_path)
height, width, _ = image.shape

rect_top_left = (0, 0)
rect_bottom_right = (width, 30)

# Define the position and size for the rectangle and text
text =  f"{predicted_class}: {confidence*100:.2f}%"
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
color = (255, 255, 255)
thickness = 2

# Draw a solid rectangle behind the text
cv2.rectangle(image, rect_top_left, rect_bottom_right, (0, 255, 0), -1)

# Add the text on top of the rectangle
cv2.putText(image, text, (5, 25), font, font_size, color, thickness)

# Alternatively, to display with OpenCV in a window
cv2.imshow("Prediction Result", image)
cv2.waitKey(0)