import requests

url = "http://192.168.1.11:8000/api/faces/all/"

response = requests.get(url)

print(response.status_code)
print(response.json())