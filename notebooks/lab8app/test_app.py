# test_app.py
import requests

data = {
    "text": "this movie was absolutely fantastic and thrilling"
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)

print("For the text:", data["text"])
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
