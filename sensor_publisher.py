import time
import random
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import json
import datetime
# configure the credentials for cilent
client = AWSIoTMQTTClient("raspberry")
client.configureEndpoint("a1ezbd0l3p8fzz-ats.iot.us-east-2.amazonaws.com", 8883)
client.configureCredentials("./root-CA.crt", "./Raspberry.private.key", "./Raspberry.cert.pem")

# Connect to AWS IoT
client.connect()

# Publish virtual sensor data to MQTT topic
# Data will be automatically stored in DynamoDB
while True:
    #timestamp = str(datetime.datetime.now())
    timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    # payload = {
    #     "temperature": random.uniform(-50, 50),
    #     "humidity": random.uniform(0, 100),
    #     'timestamp': timestamp
    # }
    # the sensor range is from jena_climate_dataset
    payload = {
        "temperature": random.uniform(-30, 40),
        "pressure": random.uniform(970, 1040),
        "relative_humidity": random.uniform(0, 100),
        "vapor_pressure": random.uniform(0, 25),
        "wind_speed": random.uniform(0, 25),
        "airtight": random.uniform(0, 1),
        "timestamp": timestamp
    }

    client.publish("raspberry/1/data", json.dumps(payload), 1)
    # publish the sensor data per 10 min
    time.sleep(60)
