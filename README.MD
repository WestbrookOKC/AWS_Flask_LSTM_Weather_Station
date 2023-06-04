# Design and Implementation of a Cloud-Based IoT System for Weather Monitoring and Prediction


climate.csv can be downloaded from: https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

static folder is used to contain the pictures which will be used to display on the website

templates folder is used to contain the html files

result folder is used to contain the pictures which i posted on the report paper


app.py is the main file that can run the website server.

Go to http://127.0.0.1:5000/LSTM or http://127.0.0.1:5000/lr, it will jump to the website
display the current sensors data and predicted value using different models.

climate.csv is the jena_climate_2009_2016 dataset. It can also be downloaded at:
https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

lr_model.py is used to train and test the linear regression model, finally,
the model is saved into model2.pkl

LSTM.py is used to train and test the LSTM model,similarly, the model
is saved into LSTM_Model.pth

modelCompare.py is used to compare the performance of two model, which will
eventually show the plot with true value, predicted value of lstm, and predicted value of linear regression.

Raspberry.cert.pem & Raspberry.private.key & Raspberry.public.key & 
Raspberry-Policy & Raspberry_accessKeys.csv & root-CA.crt are the keys provided by AWS, which
can help us get the access to AWS IOT.

receive_data_db.py is used to connect to the DynamoDB and extract the data from the database.

sensor_publisher.py is used to publish the message to the AWS via MTQQ topic

test.py is used to debug when i was programming this project

How to run this project:

1. Install all the packages needed for this project

2. There are already some data in the DynamoDB, but if you want to publish more, run sensor_publisher.py, sensors data will be sent every 1 minute.

3. run the app.py, and go to http://127.0.0.1:5000/LSTM or http://127.0.0.1:5000/lr, which will display the current sensors data and predicted value using different models




