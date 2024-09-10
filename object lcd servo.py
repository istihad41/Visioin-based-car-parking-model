import cv2
import numpy as np
import urllib.request
from time import sleep
import serial

net = cv2.dnn.readNet('D:/New folder/yolov3.weights', 'D:/New folder/yolov3.cfg')
net_1 = cv2.dnn.readNet('D:/New folder/yolov3.weights', 'D:/New folder/yolov3.cfg')

labelsPath = 'D:/New folder/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

port = 'COM11'
baud_rate = 9600
ser = serial.Serial(port, baud_rate)

def open_gate():
    command = b'o'
    ser.write(command)
    sleep(0.015)

def close_gate():
    command = b'c'
    ser.write(command)
    sleep(0.015)

def send_to_lcd1(message):
    message = message[:16]
    message = message.ljust(16)
    ser.write(message.encode())

def send_to_lcd2(message):
    message = message[:16]
    message = message.ljust(16)
    ser.write(b'm')
    ser.write(message.encode())


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

layer_names_1 = net_1.getLayerNames()
output_layers_1 = [layer_names_1[i - 1] for i in net_1.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(LABELS), 3))

font = cv2.FONT_HERSHEY_PLAIN
url = 'http://192.168.1.51/cam-mid.jpg'
cap = cv2.VideoCapture(url)
x_1, y_1, w_1, h_1 = 2, 2, 249, 474
x_2, y_2, w_2, h_2 = 380, 5, 257, 474
parking_space = 3

while True:
    response = urllib.request.urlopen(url)
    video_array = np.array(bytearray(response.read()), dtype=np.uint8)
    frame = cv2.imdecode(video_array, -1)

    roi = frame[y_1:y_1 + h_1, x_1:x_1 + w_1]
    roi_1 = frame[y_2:y_2 + h_2, x_2:x_2 + w_2]

    height, width = roi.shape[:2]
    height_1, width_1 = roi_1.shape[:2]

    blob = cv2.dnn.blobFromImage(roi, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    blob_1 = cv2.dnn.blobFromImage(roi_1, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    net_1.setInput(blob_1)
    outs_1 = net_1.forward(output_layers_1)

    class_ids = []
    confidences = []
    boxes = []
    object_counts = {}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    object_counts = {'car': 0}

    for i in range(len(boxes)):
        if i in indexes:
            if int(LABELS[class_ids[i]] == 'car'):
                x, y, w, h = boxes[i]
                label = str(LABELS[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(roi, (x, y), (x + w, y + h), color, 2)
                cv2.putText(roi, label + " " + str(round(confidence * 100, 2)), (x, y - 5), font, 1, color, 2)
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1

    for label, count in object_counts.items():
        cv2.putText(frame, f"{label}: {count}", (5, 20 * len(object_counts) + 10), font, 1, (0, 255, 0), 2)

    class_ids_1 = []
    confidences_1 = []
    boxes_1 = []
    object_counts_1 = {}

    for out_1 in outs_1:
        for detection_1 in out_1:
            scores_1 = detection_1[5:]
            class_id_1 = np.argmax(scores_1)
            confidence_1 = scores_1[class_id_1]
            if confidence_1 > 0.5:
                center_x_1 = int(detection_1[0] * width_1)
                center_y_1 = int(detection_1[1] * height_1)
                w_3 = int(detection_1[2] * width_1)
                h_3 = int(detection_1[3] * height_1)
                x_3 = int(center_x_1 - w_3 / 2)
                y_3 = int(center_y_1 - h_3 / 2)
                boxes_1.append([x_3, y_3, w_3, h_3])
                confidences_1.append(float(confidence_1))
                class_ids_1.append(class_id_1)

    indexes_1 = cv2.dnn.NMSBoxes(boxes_1, confidences_1, 0.5, 0.4)

    object_counts_1 = {'car': 0}

    for i in range(len(boxes_1)):
        if i in indexes_1:
            if int(LABELS[class_ids_1[i]] == 'car'):
                x_3, y_3, w_3, h_3 = boxes_1[i]
                label = str(LABELS[class_ids_1[i]])
                confidence = confidences_1[i]
                color = colors[class_ids_1[i]]
                cv2.rectangle(roi_1, (x_3, y_3), (x_3 + w_3, y_3 + h_3), color, 2)
                cv2.putText(roi_1, label + " " + str(round(confidence * 100, 2)), (x_3, y_3 - 5), font, 1, color, 2)
                if label in object_counts_1:
                    object_counts_1[label] += 1
                else:
                    object_counts_1[label] = 1

    for label, count_1 in object_counts_1.items():
        cv2.putText(frame, f"{label}: {count_1}", (310, 20 * len(object_counts_1) + 10), font, 1, (0, 255, 0), 2)

    car_detected = False
    for label, count in object_counts.items():
        for label, count_1 in object_counts_1.items():
            space = parking_space - count_1
            if count == 1 and space > 0:
                car_detected = True
                break

    if car_detected:
        open_gate()
    else:
        close_gate()

    for label, count_1 in object_counts_1.items():
        space = parking_space - count_1
        send_to_lcd1('Available Space:')
        send_to_lcd2(str(space))
        
    cv2.rectangle(frame, (x_1, y_1), (x_1 + w_1, y_1 + h_1), (0, 255, 0), 2)
    cv2.rectangle(frame, (x_2, y_2), (x_2 + w_2, y_2 + h_2), (0, 255, 0), 2)
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
