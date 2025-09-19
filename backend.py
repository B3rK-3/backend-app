# script that processes an uploaded image

from flask import Flask, request, jsonify
from uuid import uuid4
import jwt
import base64
from PIL import Image
from rembg.handle import handle
import torch
import clip
import datetime
from numpy import float32
import os
import psycopg2
from dotenv import load_dotenv
import geminiAPI
import hashlib
from io import BytesIO

load_dotenv()

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)


def getJWTPayload(jwtString: str) -> dict:
    """get jwt payload"""
    payload = jwt.decode(jwtString, os.environ.get("JWTSECRET"), algorithms=["HS256"])
    return payload


def validJWT(payload: dict) -> bool:
    """Check if jwt is still valid"""
    timeNow = datetime.datetime.now(datetime.timezone.utc).timestamp()
    if payload["exp"] <= timeNow:
        return False
    else:
        return True


def get_conn():
    return psycopg2.connect(
        database="postgres",
        user=os.environ["user"],
        password=os.environ["pass"],
        host=os.environ["host"],
        port=os.environ["port"],
    )


def validImageFeature(features: dict) -> bool:
    """{{
      "color_primary": "string",
      "material": "string",
      "pattern": "string",
      "tags": ["string"]
    }}"""
    featureKeys = features.keys()
    # all they keys exist
    for key in ("color_primary", "material", "pattern", "tags"):
        if key not in featureKeys:
            return False
    # the list has elements
    if not features["tags"]:
        return False
    return True


def hashStr(text: str):
    """
    Hashes a given text for database

    Parameters:
        text (str): text
    Returns:
        out (str): hashed text
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def getUpdateRefreshToken(userID: str, cursor):
    """
    make a new refreshToken and update the postgres database
    """
    token = str(uuid4())
    tokenHash = hashStr(token)

    time_now = datetime.datetime.now(datetime.timezone.utc)
    # 90 days after creation
    expiry = time_now + datetime.timedelta(days=90)

    cursor.execute(
        """
    INSERT INTO refresh_tokens (user_id, token_hash, exp)
    VALUES (%s,%s,%s)
    ON CONFLICT (user_id)
    DO UPDATE SET token_hash = EXCLUDED.token_hash,
    exp = EXCLUDED.exp""",
        (userID, tokenHash, expiry),
    )

    return token


def isValidRefreshToken(userID: str, refreshToken: str, cursor):
    tokenHash = hashStr(refreshToken)
    cursor.execute(
        """SELECT token_hash FROM refresh_tokens WHERE user_id = %s""", userID
    )
    dbTokenHash = cursor.fetchone()[0]
    if tokenHash != dbTokenHash:
        return False
    else:
        return True


def newJWT(userID: str):
    """
    Return a jwt string

    Parameters:
        userID (str): UUIDv7 hex string
    Returns:
        out (str): encoded jwt string header.payload.signature
    """
    time_now = datetime.datetime.now(datetime.timezone.utc)
    # a day after creation
    expiry = time_now + datetime.timedelta(days=1)
    payload = {
        "iss": "python-backend",
        "sub": userID,
        "iat": time_now,
        "exp": expiry,
    }
    jwtString = jwt.encode(payload, os.environ.get("JWTSECRET"), algorithm="HS256")
    return jwtString


def clipImage(bytesObj: bytes):
    """
    takes in a bytes object and returns a vector.

    Parameters:
        bytesObj (bytes): The input image.

    Returns:
        List[int]: a list of size 512 containing the vector.
    """
    image = preprocess(Image.open(BytesIO(bytes))).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features_normalized = image_features / image_features.norm(
            dim=-1, keepdim=True
        )  # normalize
    embedding: list = (
        image_features_normalized.cpu().numpy().flatten().astype(float32).tolist()
    )
    return embedding


@app.route("/pushdb", methods=["POST"])
def pushdb():
    payload = request.get_json()
    jwtString: str = payload["jwt"]
    img: str = payload["img"]  # base64 string
    garment_type: str = payload["type"]

    jwtPayload = getJWTPayload(jwtString)
    if not validJWT(jwtPayload):
        return jsonify(
            {"status": "ERROR", "ERROR": "expired_jwt", "message": "JWT EXPIRED"}
        ), 401
    userID = jwtPayload["sub"]

    conn = get_conn()
    cursor = conn.cursor()

    bytesIn = base64.b64decode(img)
    print("rembg")
    bytesOut = handle(bytesIn)  # remove bg

    open("req.jpg", "wb").write(bytesOut)
    embedding = clipImage(bytesOut)

    features = geminiAPI.generate_tags(
        bytesIn, garment_type=garment_type, image_format="jpeg"
    )
    if not validImageFeature(features):
        return jsonify(
            {
                "status": "ERROR",
                "message": "GEMINI RETURNED BAD FEATURES",
                "ERROR": "internal_error",
            }
        )

    cursor.execute(
        "INSERT INTO garments (user_id,garment_type,image_url,color_primary,material,pattern,tags) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id;",
        (
            userID,
            garment_type,
            "something",
            features["color_primary"],
            features["material"],
            features["pattern"],
            features["tags"],
        ),
    )
    garment_id = str(cursor.fetchone()[0])
    cursor.execute("INSERT INTO garment_embed VALUES (%s, %s)", (garment_id, embedding))
    conn.commit()
    conn.close()
    return jsonify(
        {
            "status": "SUCCESS",
            "message": "UPLOADED IMAGE SUCCESSFULLY RETURN IMAGE_ID",
            "image_id": garment_id,
        }
    ), 201


@app.route("/register", methods=["POST"])
def register():
    payload = request.get_json(force=True)
    email = payload["email"]
    password = payload["password"]
    hashedPass = hashStr(password)

    conn = get_conn()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (email, hashed_pass) VALUES (%s, %s) RETURNING id;",
                (email, hashedPass),
            )
            userID = str(cursor.fetchone()[0])

            # rotate/issue refresh token for user
            refreshToken = getUpdateRefreshToken(userID, cursor)
            # short-lived access JWT
            jwtString = newJWT(userID)

        conn.commit()
        return jsonify(
            {
                "status": "SUCCESS",
                "refreshToken": refreshToken,  # set as HttpOnly cookie in real app
                "jwt": jwtString,
                "message": "RETURNED REFRESHTOKEN, JWT",
            }
        ), 201

    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return jsonify(
            {"status": "ERROR", "ERROR": "email_exists", "message": "EMAIL NOT UNIQUE"}
        ), 409

    finally:
        conn.close()


@app.route("/updatejwt", methods=["POST"])
def updateJWT():
    payload = request.get_json()
    jwtString = payload["jwt"]
    refreshToken = payload["refreshToken"]

    conn = get_conn()
    cursor = conn.cursor()

    jwtPayload = getJWTPayload(jwtString)
    userID = jwtPayload["sub"]

    validRefreshToken = isValidRefreshToken(userID, refreshToken, cursor)
    if not validRefreshToken:
        return jsonify(
            {
                "status": "ERROR",
                "ERROR": "invalid_refresh_token",
                "message": "INVALID REFRESH TOKEN",
            }
        ), 401

    newJwtString = newJWT(userID)
    conn.close()
    return jsonify(
        {
            "status": "SUCCESS",
            "jwt": newJwtString,
            "message": "RETURNED JWT",
        }
    ), 201


# refreshes the reshreshToken
@app.route("/login", methods=["POST"])
def login():
    payload = request.get_json()
    password = payload["password"]
    email = payload["email"]

    conn = get_conn()
    cursor = conn.cursor()

    hashedPass = hashStr(password)
    cursor.execute(
        "SELECT id FROM users WHERE email = %s AND hashed_pass = %s",
        (email, hashedPass),
    )
    user = cursor.fetchone()
    if not user:
        return jsonify(
            {
                "status": "ERROR",
                "message": "USER DOES NOT EXIST",
                "ERROR": "no_such_user",
            }
        ), 401

    userID = user[0]
    refreshToken = getUpdateRefreshToken(userID, cursor)
    jwtString = newJWT(userID)
    conn.commit()
    conn.close()
    return jsonify(
        {
            "status": "SUCCESS",
            "refreshToken": refreshToken,
            "jwt": jwtString,
            "message": "RETURNED REFRESHTOKEN AND JWTTOKEN",
        }
    ), 201


"""WITH filtered AS (
  SELECT id
  FROM garments
  WHERE user_id = $1
    AND garment_type = 'dress'
    AND color_primary = 'navy'
    AND (tags @> '["vneck"]'::jsonb)
)
SELECT g.id, g.image_url, ge.embedding <-> $2::vector AS dist
FROM garments g
JOIN garment_embed ge ON ge.garment_id = g.id
JOIN filtered f ON f.id = g.id
ORDER BY dist
LIMIT 20;"""
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
