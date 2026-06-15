import urllib.request
import json
import mimetypes

def post_file(url, file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
    headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
    
    body = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8')
    body += file_data
    body += f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return str(e)

file_path = r"C:\Users\Andriana  Musonza\.gemini\antigravity\scratch\ml_backend\myeloma_images\PCMMD Plasma Cells for Multiple Myeloma Diagnosis\data\detection\train\images\1692243677854.jpg"
print("Myeloma response:", post_file("http://127.0.0.1:5001/predict_myeloma", file_path))

file_path2 = r"C:\Users\Andriana  Musonza\.gemini\antigravity\scratch\Leukemia_images\training_data\fold_0\all\UID_11_10_1_all.bmp"
print("Blood response:", post_file("http://127.0.0.1:5001/predict_blood", file_path2))

