import requests

response = requests.get('http://127.0.0.1:9000/FaceDetection/test')
data_name = response.json()
print(data_name)