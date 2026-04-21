import requests

url = "https://prediction-api-330013579477.europe-west1.run.app/predict"

# Full 24-feature payload matching the PatientData schema
payload = {
    # Numeric features
    "age": 48.0,
    "bp": 80.0,
    "bgr": 121.0,
    "bu": 36.0,
    "sc": 1.2,
    "sod": 137.0,
    "pot": 4.5,
    "hemo": 15.4,
    "pcv": 44.0,
    "wc": 7800.0,
    "rc": 5.2,
    # Categorical features
    "sg": "1.020",
    "al": "1",
    "su": "0",
    "rbc": "normal",
    "pc": "normal",
    "pcc": "notpresent",
    "ba": "notpresent",
    "htn": "no",
    "dm": "no",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no"
}

response = requests.post(url, json=payload)
print("Status:", response.status_code)
print("Response:", response.json())
