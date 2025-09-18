# script that processes an uploaded image

from flask import Flask, request
import base64
from PIL import Image
from rembg.handle import handle
import torch
import clip
from PIL import Image
import datetime
import numpy as np
import os

import psycopg2

app = Flask(__name__)


def clipImage(bytesObj: bytes):
    """
    takes in a bytes object and returns a vector.

    Parameters:
        bytesObj (bytes): The input image.

    Returns:
        List[int]: a list of size 512 containing the vector.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open(bytesObj)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features_normalized = image_features / image_features.norm(
            dim=-1, keepdim=True
        )  # normalize
    embedding: list = (
        image_features_normalized.cpu().numpy().flatten().astype(np.float32).tolist()
    )
    return embedding


@app.route("/pushdb", methods=["POST"])
def pushdb():
    payload = request.get_json()
    userid: str = payload["userid"]
    img: str = payload["img"]

    bytesIn = base64.b64decode(img)
    bytesOut = handle(bytesIn)
    embedding = clipImage(bytesOut)

    conn = psycopg2.connect(
        database="postgres",
        user=os.environ["user"],
        password=os.environ["pass"],
        host=os.environ["host"],
        port=os.environ["port"],
    )
    cursor = conn.cursor()

    cursor.execute(
        f"INSERT INTO {table} (userid,garment_type,,embed) VALUES (%s,%s)"
    )
