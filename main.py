from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

net = cv2.dnn.readNetFromDarknet("models/customWA-test.cfg", "models/customWA_final.weights")

classes = ['bear', 'bison', 'cat', 'chimpansee', 'cow', 'deer', 'dog', 'donkey', 'elephant', 'fox', 'goat',
           'gorilla', 'horse', 'hyena', 'leopard', 'lion', 'panda', 'pig', 'reindeer', 'rhinoceros', 'sheep',
           'tiger', 'wolf', 'zebra']

def perform_object_detection(image):
    h, w, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, width, height = boxes[i]
            label = classes[class_ids[i]]
            confidence = round(confidences[i], 2)
            detected_objects.append({'label': label, 'confidence': confidence, 'box': [x, y, width, height]})

    return detected_objects

@app.route('/')
def index():
    return "Hello, Welcome to Wild Animal API"

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Receive image from frontend
    image_file = request.files['image']
    nparr = np.fromstring(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Perform object detection
    detected_objects = perform_object_detection(image)

    # Return detected objects
    return jsonify(detected_objects)

if __name__ == '__main__':
    app.run(debug=True)
