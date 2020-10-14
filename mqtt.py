import paho.mqtt.client as mqtt
import time
def on_log(client, userdata, level, buf):
    print("Log: "+buf)

def on_connect(client, userdata, flags, rc):
    if rc==0:
        print("Connected OK")
    else:
        print("Bad connection returned code=" , rc)
def on_disconnect(client, userdata, flags, rc):
    print("Disconnected result code "+str(rc))
          
def on_message(client, userdata, msg):
    topic = msg.topic
    m_decode = str(msg.payload.decode("utf-8", "ignore"))
    print("Message received", m_decode)
          
broker = "192.168.43.62"

client = mqtt.Client("python1")
client.on_connect = on_connect
client.on_connect = on_disconnect          
client.on_log = on_log
client.on_message = on_message
print("Connecting to Broker", broker)
client.connect(broker)
client.loop_start()
client.subscribe("car/sensor1")                                    
time.sleep(40)
client.loop_stop()
client.disconnect()
