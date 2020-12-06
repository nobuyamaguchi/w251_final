import paho.mqtt.client as mqtt
import numpy as np 
import cv2 as cv2
import sys
import os
import boto3

LOCAL_MQTT_HOST = "172.18.0.1"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC = "xavierphotos" 

global i 
i =1

def on_connect(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    
def on_message(client, userdata, msg):
    
    print("message received!")
    print("type: ", type(msg))

    global i 
    i += 1

    p = np.fromstring(msg.payload, np.uint8)
    print("type converted: ", type(p))
    img = cv2.imdecode(p, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    cv2.imwrite("mybucket/image_" + str(i) + ".png", img)

    file_name = "image_" + str(i) + ".png"
    img_file = "mybucket/" + file_name
    bucket_name = "w251finalpj"

    s3_client = boto3.client('s3', aws_access_key_id="AKIAJSBCM5QX2L5XLACA", aws_secret_access_key="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" )
    #response = s3_client.upload_file(img_file, bucket_name, file_name)
    s3_client.upload_file(img_file, bucket_name, file_name)

local_mqttclient = mqtt.Client(client_id="imageprocessor")
local_mqttclient.on_connect = on_connect
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.subscribe(LOCAL_MQTT_TOPIC)
local_mqttclient.on_message = on_message

local_mqttclient.loop_forever()