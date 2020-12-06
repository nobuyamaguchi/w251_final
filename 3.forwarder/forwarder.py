import paho.mqtt.client as mqtt

print("python started running")

# set parameters for xavier
LOCAL_MQTT_HOST = "172.20.0.1" #"mosquitto"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC ="xavierphotos"

# set parameters for aws
REMOTE_MQTT_HOST = "ec2-18-224-72-125.us-east-2.compute.amazonaws.com" #"mosquitto" 
REMOTE_MQTT_PORT = 1883
REMOTE_MQTT_TOPIC = "xavierphotos"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))

def on_connect_remote(client, userdata, flags, rc):
        print("connected to remote broker with rc: " + str(rc))

def on_message(client,userdata, msg):
  try:
    print("message received!")
    msg = msg.payload
    remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
    print("message published to remote !", REMOTE_MQTT_HOST, REMOTE_MQTT_TOPIC)
  except:
    print("Unexpected error:", sys.exc_info()[0])

# xavier 
local_mqttclient = mqtt.Client(client_id="forwarder_local")
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.subscribe(LOCAL_MQTT_TOPIC)
local_mqttclient.on_message = on_message


# aws 
remote_mqttclient = mqtt.Client(client_id="forwarder_remote")
remote_mqttclient.loop_start()
remote_mqttclient.on_connect = on_connect_remote
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)


local_mqttclient.loop_forever()


