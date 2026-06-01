from decouple import config
import requests

print(config("DJANGO_API_URL"))

url = f"{config('DJANGO_API_URL')}/faces/all/"

try:

    response = requests.get(url, timeout=30)

    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
except Exception as e:

    print(f"Error: {str(e)}")