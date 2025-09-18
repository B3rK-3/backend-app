# Image Processing Backend

This repository contains a Python Flask backend server that provides a single API endpoint for processing user-uploaded images. The server integrates **CLIP**, **U²-Net**, **Rembg**, and the **Gemini API** to remove backgrounds, extract semantic tags, and push processed data to a database for later use.

---

## 🚀 Features

- **Single Endpoint**: `/pushdb` (POST)
- **Input**: JSON payload of the form:
  ```json
  {
    "img": "<base64 string>",
    "userid": "<string>"
  }
