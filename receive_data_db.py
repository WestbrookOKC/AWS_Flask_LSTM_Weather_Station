import boto3

# Return the sensor data in DynamoDB
def get_sensor_data():
    dynamodb = boto3.resource('dynamodb', aws_access_key_id='xxx',
                              aws_secret_access_key='xxx', region_name='us-east-2')
    table = dynamodb.Table('raspberry_data')
    response = table.scan()
    items = response['Items']

    data = []
    for item in items:
        data.append(item['payload'])

    # Sort the data based on the timestamp before return
    data = sorted(data, key=lambda x: x['timestamp'], reverse=True)
    return data

