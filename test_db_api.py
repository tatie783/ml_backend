import requests
import json

base_url = "http://127.0.0.1:5001"

# Test Save
payload = {
    'hospital_no': 'TEST-999',
    'age': 22,
    'dob': '2002-02-02',
    'address': 'Testing Lane',
    'test_type': 'BLOOD',
    'prediction': 'Normal',
    'confidence': '99.9%'
}

print("Testing POST /save_record...")
try:
    r_save = requests.post(f"{base_url}/save_record", json=payload)
    print(f"Status: {r_save.status_code}")
    print(f"Response: {r_save.text}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting GET /get_history...")
try:
    r_hist = requests.get(f"{base_url}/get_history")
    print(f"Status: {r_hist.status_code}")
    print(f"Response: {r_hist.text}")
except Exception as e:
    print(f"Error: {e}")
