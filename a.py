import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("C:/internship project/Autonomous Vehicles/yolov3.weights", "C:/internship project/Autonomous Vehicles/yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Specify the output layer indices
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

if len(unconnected_layers) > 0:
    output_layers = []
    for i in range(len(unconnected_layers)):
        if isinstance(unconnected_layers[i], list) and len(unconnected_layers[i]) > 0:
            index = unconnected_layers[i][0] - 1
            if index >= 0 and index < len(layer_names):
                output_layers.append(layer_names[index])
    print("Output layers:")
    for output_layer in output_layers:
        print(output_layer)
else:
    print("No unconnected output layers found.")

# Load classes
classes = []
with open("C:/internship project/Autonomous Vehicles/coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Set up video capture
cap = cv2.VideoCapture("C:\internship project\Autonomous Vehicles\Y2meta.app-Mere Bina _ Crook _ Night Drive _ Highway _ Car driving status _-(1080p60).mp4")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Forward pass through YOLO network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:  # Adjust confidence threshold
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Draw bounding box around the object
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
