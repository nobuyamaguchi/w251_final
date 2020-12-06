import paho.mqtt.client as mqtt
import time
import cv2
import numpy as np
import time

import glob

dispW=800  
dispH=600  

LOCAL_MQTT_HOST= "172.20.0.1" #"mosquitto" #"172.18.0.2"
LOCAL_MQTT_PORT=1883 
LOCAL_MQTT_TOPIC="xavierphotos"

#MQTT 

def on_publish(client,userdata,result):           
    print("data published to aws remote topic  \n")
    pass

def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))


# added 0912 1205
dc_flag = False

def on_disconnect(client, userdata,  rc):
    global dc_flag
    dc_flag = True
    if dc_flag:
        client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)      


local_mqttclient = mqtt.Client(client_id="detector")
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_publish = on_publish

local_mqttclient.on_disconnect = on_disconnect

# Xavier Camera section
cam = cv2.VideoCapture('/dev/video0')

# added 11.27 to make the process faster
cam.set(cv2.CAP_PROP_BUFFERSIZE, 2) 
cam.set(cv2.CAP_PROP_FPS, 2)   

cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
W=cam.get(cv2.CAP_PROP_FRAME_WIDTH)
H=cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

### Loading face cascade
face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml') # changed opencv to opencv4 11.22


### Yolo Gun detector set up
# Load Yolo
net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["Gun"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))



while True:
    ret,frame = cam.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # decided not using gray to match the shape of dnn
   
    # Loading image
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)  #11.26 replaced img with gray >> frame   
 
    # height, width, channels = img.shape   
    height, width = img.shape[:2]


    ##### Detecting Gun with Yolo
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    start = time.time()

    net.setInput(blob)
    outs = net.forward(output_layers)

    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # Showing Gun informations on the screen
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
                print("Gun detected")
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
    # print(indexes) # hided on 11.26 
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 20), font, 2, color, 2)   #20 instead of 30, 2 instead of 3, 1 instead of 2
            gun_cropped_frame = img[y:y+h, x:x+w]
            rc,gun_png = cv2.imencode('.png', gun_cropped_frame)
            gun_msg = gun_png.tobytes()


	    ##### detecting face
            faces = face_cascade.detectMultiScale(img, 1.3, 5)

            for (x,y,w,h) in faces:
                face = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
                face_cropped_frame = img[y:y+h, x:x+w]
                rc,face_png = cv2.imencode('.png', face_cropped_frame)
                face_msg = face_png.tobytes()
		# print(msg) # hided on 11.26

                local_mqttclient.publish(LOCAL_MQTT_TOPIC, face_msg, qos=0, retain=False)  # "ret=" removed
                print("Face cropped image published")

            local_mqttclient.publish(LOCAL_MQTT_TOPIC, gun_msg, qos=0, retain=False)  # "ret=" removed
            print("Gun cropped image published")
            

    cv2.imshow('myCam', img)   #used img instead of frame    
    cv2.moveWindow('myCam',100,100)

    if cv2.waitKey(10)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


