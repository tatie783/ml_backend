import requests

url = "http://127.0.0.1:5000/predict_blood"
file_path = r"C:\Users\Andriana  Musonza\.gemini\antigravity\scratch\Leukemia_images\training_data\fold_0\all\UID_11_10_1_all.bmp"

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print(response.json())
