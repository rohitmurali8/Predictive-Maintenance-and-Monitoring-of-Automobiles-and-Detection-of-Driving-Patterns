import paho.mqtt.client as mqtt
client = mqtt.Client()
client.connect("192.168.43.62",1883,60)
client.publish("car/sensor1","BAROMETRIC PRESSURE - 95.34")
client.disconnect()
