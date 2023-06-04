import numpy as np
from flask import Flask, render_template,request
import pickle
from receive_data_db import get_sensor_data
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import torch
from LSTM import LSTM
import time
#Initialize the flask App

app = Flask(__name__)
# Initialize  the parameter for lstm model
input_size = 6
hidden_size = 64
output_size = 1
model_LSTM = LSTM(input_size, hidden_size, output_size)
# load model
model_LSTM.load_state_dict(torch.load('LSTM_Model.pth'))
model = pickle.load(open('model2.pkl', 'rb'))
# extract data from database
sensor_data = get_sensor_data()

# this page is used to debug
@app.route('/',methods=['GET','POST'])
def predict():
    if request.method =="POST":
        # put application's code here
        int_features = [float(x) for x in request.form.values()]

        final_features = [int_features]

        prediction = model.predict(final_features)
        # output = prediction  # round(prediction[0], 2)
    else:
        prediction = ''
    return render_template('index.html', prediction_text='Temperature is  :{}'.format(prediction))


# display of linear regression model
@app.route('/lr')
def raspberry():
    # round all sensor data into 2
    temperature = float(round(sensor_data[0]['temperature'], 2))
    pressure = float(round(sensor_data[0]['pressure'], 2))
    humidity = float(round(sensor_data[0]['relative_humidity'], 2))
    vapor_pressure = float(round(sensor_data[0]['vapor_pressure'], 2))
    wind_speed = float(round(sensor_data[0]['wind_speed'], 2))
    airtight = float(round(sensor_data[0]['airtight'], 2))
    predicted_timestamp = sensor_data[0]['timestamp']

    # Get the current time in local time zone
    local_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(local_tz)
    # convert datetime object
    # for plot clear
    dt_obj = datetime.strptime(predicted_timestamp, '%Y-%m-%d %H:%M')

    # Add 2 hours 
    dt_obj += timedelta(hours=2)

    # Convert the datetime object  local time zone
    dt_obj = local_tz.localize(dt_obj)

    # Format the datetime 
    predicted_timestamp = dt_obj.strftime('%Y-%m-%d %H:%M')
    # model prediction
    predicted_value = round(model.predict([[temperature,pressure,humidity,vapor_pressure,wind_speed,airtight]])[0], 2)

    # Save the predicted plot and get the filename
    filename = save_predicted_plot(predicted_timestamp, predicted_value)
    print(filename)
    items = {
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'vapor_pressure': vapor_pressure,
        'wind_speed': wind_speed,
        'airtight': airtight,
        'predicted_value': predicted_value,
        'plot_filename': filename
    }

    return render_template('predict_output.html', items=items)


def save_predicted_plot(predicted_timestamp, predicted_value):
    # Get temperature and timestamp data
    temperature_data = [float(round(d['temperature'], 2)) for d in reversed(sensor_data)]
    # only display lastest 5 data
    temperature_data = temperature_data[-5:]
    timestamp_data = [d['timestamp'] for d in reversed(sensor_data)]
    timestamp_data = timestamp_data[-5:]

    # Format timestamps to only show hour and minute
    timestamp_data = [datetime.strptime(t, '%Y-%m-%d %H:%M').strftime('%H:%M') for t in timestamp_data]
    predicted_timestamp = datetime.strptime(predicted_timestamp, '%Y-%m-%d %H:%M').strftime('%H:%M')

    # Plot history, model prediction, and true value
    plt.plot(timestamp_data, temperature_data)
    plt.plot(predicted_timestamp, predicted_value, "rX")
    plt.legend(["History", "Model Prediction"])
    plt.xlabel("Time")
    plt.ylabel("Temperature (in degC)")
    #plt.show()
    filename = f"static/{predicted_timestamp}.png"

    plt.savefig(filename)

    # Close plot
    plt.close()

    return filename

@app.route('/LSTM')
def raspberry_LSTM():
    # round data into 2
    temperature = float(round(sensor_data[0]['temperature'], 2))
    pressure = float(round(sensor_data[0]['pressure'], 2))
    humidity = float(round(sensor_data[0]['relative_humidity'], 2))
    vapor_pressure = float(round(sensor_data[0]['vapor_pressure'], 2))
    wind_speed = float(round(sensor_data[0]['wind_speed'], 2))
    airtight = float(round(sensor_data[0]['airtight'], 2))
    predicted_timestamp = sensor_data[0]['timestamp']

    # Get the current time in local time zone
    local_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(local_tz)

    # convert it to a datetime object
    dt_obj = datetime.strptime(predicted_timestamp, '%Y-%m-%d %H:%M')

    # Add 2 hours to the datetime object
    dt_obj += timedelta(hours=2)

    dt_obj = local_tz.localize(dt_obj)

    # Format the datetime
    predicted_timestamp = dt_obj.strftime('%Y-%m-%d %H:%M')
    data = sensor_data[0]
    #predicted_value = round(model.predict([[temperature, humidity]])[0], 2)


    # lstm prediction
    predicted_temperature = get_LSTM_result(data)
    # print(predicted_temperature.item())
    predicted_value = round(predicted_temperature.item(), 2)
    # Save the predicted plot and get the filename
    filename = save_predicted_plot2(predicted_timestamp, predicted_value)
    # Give time to store the img, since lstm run slowly
    time.sleep(5)
    print(filename)
    items = {
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'vapor_pressure': vapor_pressure,
        'wind_speed': wind_speed,
        'airtight': airtight,
        'predicted_value': predicted_value,
        'plot_filename': filename
    }

    return render_template('predict_LSTM.html', items=items)

def get_LSTM_result(data):
    test_data = np.array([[[data['temperature'], data['pressure'], data['relative_humidity'], data['vapor_pressure'],
                            data['wind_speed'], data['airtight']]]], dtype=np.float32)
    x_test = torch.from_numpy(test_data).float()
    with torch.no_grad():
        model_LSTM.eval()
        predicted_temperature = model_LSTM(x_test)
    return predicted_temperature

def save_predicted_plot2(predicted_timestamp, predicted_value):
    # Get temperature and timestamp data
    temperature_data = [float(round(d['temperature'], 2)) for d in reversed(sensor_data)]
    temperature_data = temperature_data[-5:]
    timestamp_data = [d['timestamp'] for d in reversed(sensor_data)]
    timestamp_data = timestamp_data[-5:]

    # Format timestamps to only show hour and minute
    timestamp_data = [datetime.strptime(t, '%Y-%m-%d %H:%M').strftime('%H:%M') for t in timestamp_data]
    predicted_timestamp = datetime.strptime(predicted_timestamp, '%Y-%m-%d %H:%M').strftime('%H:%M')

    # Plot history, model prediction, and true value
    plt.plot(timestamp_data, temperature_data)
    plt.plot(predicted_timestamp, predicted_value, "rX")
    plt.legend(["History", "Model Prediction"])
    plt.xlabel("Time")
    plt.ylabel("Temperature (in degC)")
    #plt.show()
    #predicted_timestamp = predicted_timestamp + '5'
    #filename = f"static/{predicted_timestamp}.png"
    filename = f"static/{predicted_timestamp.replace(':', '_')}.png"
    plt.savefig(filename)

    # Close plot
    plt.close()

    return filename

if __name__ == '__main__':
    app.run()

