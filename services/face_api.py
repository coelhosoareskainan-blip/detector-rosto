import requests
import os

API_KEY = os.getenv("FACEPP_API_KEY")
API_SECRET = os.getenv("FACEPP_API_SECRET")

DETECT_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

def get_embedding(image_bytes):
    files = {
        "image_file": image_bytes
    }

    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_landmark": 0,
        "return_attributes": "none"
    }

    response = requests.post(DETECT_URL, data=data, files=files)
    result = response.json()

    if "faces" not in result or len(result["faces"]) == 0:
        return None

    return result["faces"][0]["face_token"]

