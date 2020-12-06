import cv2
import numpy as np
import glob
import random


# Load Yolo
net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["Gun"]

# Images path
images_path_pos = glob.glob(r"C:\Users\satos\Desktop\200829_Program_Files\MIDS\W251_Deep_Learning\Final_Project\Evaluation\pos\*.jpg")
images_path_neg = glob.glob(r"C:\Users\satos\Desktop\200829_Program_Files\MIDS\W251_Deep_Learning\Final_Project\Evaluation\neg\*.jpg")


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


######################## Running Yolo on Positive Images ########################

# Insert here the path of your images
random.shuffle(images_path_pos)
# loop through all the images

# Setting counters
count_pos = 0
true_pos = 0

for img_path in images_path_pos:
    # adding 1 four count
    count_pos += 1
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1, fy=1)  # 1 instead of 0.4
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    if len(confidences) > 0:
        true_pos += 1

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    print(f'count_pos: {count_pos}, true_pos: {true_pos}')

cv2.destroyAllWindows()

# Calculating false positive
false_neg = count_pos - true_pos


######################## Running Yolo on Negative Images ########################

# Insert here the path of your images
random.shuffle(images_path_neg)
# loop through all the images

# Setting counters
count_neg = 0
false_pos = 0

for img_path in images_path_neg:
    # adding 1 four count
    count_neg += 1
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1, fy=1)  # 1 instead of 0.4
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    if len(confidences) > 0:
        false_pos += 1

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    print(f'count_neg: {count_neg}, false_pos: {false_pos}')

cv2.destroyAllWindows()

# Calculating false negative
true_neg = count_neg - false_pos


# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)

print(f'Precision : {true_pos/(true_pos + false_pos)}')
print(f'Recall : {true_pos/(true_pos + false_neg)}')

