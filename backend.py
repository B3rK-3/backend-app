# script that processes an uploaded image

from flask import Flask, request, jsonify
from pgvector.psycopg2 import register_vector
from pgvector import Vector
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
from collections import defaultdict

load_dotenv()

app = Flask(__name__)
device = "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)


class RETURNS:
    class ERRORS:
        @staticmethod
        def bad_jwt():
            return (
                jsonify(
                    {
                        "status": "ERROR",
                        "ERROR": "expired_jwt",
                        "message": "JWT EXPIRED",
                    }
                ),
                401,
            )

        @staticmethod
        def bad_image():
            return (
                jsonify(
                    {
                        "status": "ERROR",
                        "message": "GEMINI RETURNED BAD FEATURES",
                        "ERROR": "not_clothing",
                    }
                ),
                401,
            )

        @staticmethod
        def bad_email():
            return (
                jsonify(
                    {
                        "status": "ERROR",
                        "ERROR": "email_exists",
                        "message": "EMAIL NOT UNIQUE",
                    }
                ),
                409,
            )

        @staticmethod
        def bad_refresh_token():
            return (
                jsonify(
                    {
                        "status": "ERROR",
                        "ERROR": "invalid_refresh_token",
                        "message": "INVALID REFRESH TOKEN",
                    }
                ),
                401,
            )

        @staticmethod
        def bad_login():
            return (
                jsonify(
                    {
                        "status": "ERROR",
                        "message": "USER DOES NOT EXIST",
                        "ERROR": "no_such_user",
                    }
                ),
                401,
            )

        @staticmethod
        def internal_error():
            return (
                jsonify(
                    {
                        "status": "ERROR",
                        "message": "INTERNAL SERVER ERROR",
                        "ERROR": "internal_server_error",
                    }
                ),
                500,
            )

    class SUCCESS:
        @staticmethod
        def return_garment_id(garment_id: str):
            return (
                jsonify(
                    {
                        "status": "SUCCESS",
                        "message": "UPLOADED IMAGE SUCCESSFULLY RETURN IMAGE_ID",
                        "image_id": garment_id,
                    }
                ),
                201,
            )

        @staticmethod
        def return_jwt_refresh_tokens(jwtString: str, refreshToken: str):
            return (
                jsonify(
                    {
                        "status": "SUCCESS",
                        "refreshToken": refreshToken,
                        "jwt": jwtString,
                        "message": "RETURNED REFRESHTOKEN, JWT",
                    }
                ),
                201,
            )

        @staticmethod
        def return_jwt_token(jwtString: str):
            return (
                jsonify(
                    {
                        "status": "SUCCESS",
                        "jwt": jwtString,
                        "message": "RETURNED JWT",
                    }
                ),
                201,
            )

        @staticmethod
        def return_garment_images(images: dict):
            return (
                jsonify(
                    {
                        "status": "SUCCESS",
                        "images": images,
                        "message": "RETURNED IMAGES",
                    }
                ),
                201,
            )

        @staticmethod
        def return_chat_message(text: str):
            return (
                jsonify(
                    {
                        "status": "SUCCESS",
                        "chatMessage": text,
                        "message": "RETURNED CHATBOT MESSAGE",
                    }
                ),
                201,
            )


def getJWTPayload(jwtString: str) -> dict:
    """get jwt payload"""
    try:
        payload = jwt.decode(
            jwtString, os.environ.get("JWTSECRET"), algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidSignatureError:
        return None


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
    image = preprocess(Image.open(BytesIO(bytesObj))).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features_normalized = image_features / image_features.norm(
            dim=-1, keepdim=True
        )  # normalize
    embedding: list = (
        image_features_normalized.cpu().numpy().flatten().astype(float32).tolist()
    )
    return embedding


def clipText(text: str):
    with torch.no_grad():
        textTokens = clip.tokenize(text).to(device)
        text_embedding = model.encode_text(textTokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.cpu().tolist()[0]


@app.route("/pushdb", methods=["POST"])
def pushdb():
    try:
        payload = request.get_json()
        jwtString: str = payload["jwt"]
        img: str = payload["img"]  # base64 string
        garment_type: str = payload["type"]

        jwtPayload = getJWTPayload(jwtString)
        if not validJWT(jwtPayload):
            return RETURNS.ERRORS.bad_jwt
        userID = jwtPayload["sub"]

        conn = get_conn()
        cursor = conn.cursor()

        bytesIn = base64.b64decode(img)
        print("rembg...")
        bytesOut = handle(bytesIn)  # remove bg

        embedding = clipImage(bytesOut)

        features = geminiAPI.generate_tags(
            bytesIn, garment_type=garment_type, image_format="jpeg"
        )
        if not validImageFeature(features):
            return RETURNS.ERRORS.bad_image()

        garment_id = str(uuid4())
        image_path = f"garments/{userID}/{garment_id}.jpg"
        cursor.execute(
            "INSERT INTO garments (id,user_id,garment_type,image_url,color_primary,material,pattern,tags) VALUES (%s,%s,%s,%s,%s,%s,%s, %s) RETURNING id;",
            (
                garment_id,
                userID,
                garment_type,
                image_path,
                features["color_primary"],
                features["material"],
                features["pattern"],
                features["tags"],
            ),
        )

        f = open(image_path, "wb+")
        f.write(bytesOut)
        f.close()

        cursor.execute(
            "INSERT INTO garment_embed VALUES (%s, %s)", (garment_id, embedding)
        )
        conn.commit()
        conn.close()
        return RETURNS.SUCCESS.return_garment_id(garment_id)
    except BaseException as error:
        print(error)
        return RETURNS.ERRORS.internal_error()


@app.route("/register", methods=["POST"])
def register():
    try:
        payload = request.get_json(force=True)
        email = payload["email"]
        password = payload["password"]
        hashedPass = hashStr(password)

        conn = get_conn()
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
        os.makedirs(f"garments/{userID}")
        conn.commit()
        conn.close()
        return RETURNS.SUCCESS.return_jwt_refresh_tokens(jwtString, refreshToken)
    except psycopg2.errors.UniqueViolation as error:
        print(error)
        conn.rollback()
        return RETURNS.ERRORS.bad_email()
    except BaseException as error:
        print(error)
        return RETURNS.ERRORS.internal_error()


@app.route("/updatejwt", methods=["POST"])
def updateJWT():
    try:
        payload = request.get_json()
        jwtString = payload["jwt"]
        refreshToken = payload["refreshToken"]

        conn = get_conn()
        cursor = conn.cursor()

        jwtPayload = getJWTPayload(jwtString)
        userID = jwtPayload["sub"]

        if not isValidRefreshToken(userID, refreshToken, cursor):
            return RETURNS.ERRORS.bad_refresh_token()

        newJwtString = newJWT(userID)
        conn.close()
        return RETURNS.SUCCESS.return_jwt_token(newJwtString)
    except BaseException as error:
        print(error)
        return RETURNS.ERRORS.internal_error()


# refreshes the reshreshToken
@app.route("/login", methods=["POST"])
def login():
    try:
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
            return RETURNS.ERRORS.bad_login()

        userID = user[0]
        refreshToken = getUpdateRefreshToken(userID, cursor)
        jwtString = newJWT(userID)
        conn.commit()
        conn.close()
        return RETURNS.SUCCESS.return_jwt_refresh_tokens(jwtString, refreshToken)
    except BaseException as error:
        print(error)
        return RETURNS.ERRORS.internal_error()


# gemini chat for outfit
@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json()
        conversation = payload["convo"]
        jwtString = payload["jwtString"]

        jwtPayload = getJWTPayload(jwtString)
        userID = jwtPayload["sub"]

        response = geminiAPI.getConvoResponse(conversation)

        print("Return Type:", type(response))
        print("Return:", response)
        if type(response) is dict:
            images = getGarments(response, userID)
            return RETURNS.SUCCESS.return_garment_images(images)
        elif type(response) is str:
            return RETURNS.SUCCESS.return_chat_message(response)
        raise Exception(
            "WRONGLY FORMATTED RESPONSE: " + str(response),
        )
    except BaseException as error:
        print(error)
        return RETURNS.ERRORS.internal_error()


def getGarments(jsonObj: dict, userID: str):
    """
    Parses the database and return images for the given garments
    """
    conn = get_conn()
    register_vector(conn)
    cursor = conn.cursor()

    seen = defaultdict(set)
    res = defaultdict(list)

    for garment_type in jsonObj.keys():
        for option in jsonObj[garment_type]:
            """
            option: {
    "garment_type": "string",
    "color_primary": "string",
    "material": "string",
    "pattern": "string",
    "tags": ["string"],
    "image_description": "string"
    }
            """
            garment_type = option["garment_type"]
            textEmbed = clipText(option["image_description"])
            color_primary = option["color_primary"]
            material = option["material"]
            pattern = option["pattern"]
            tags = option["tags"]
            params = {
                "user_id": userID,  # UUID
                "garment_type": garment_type,
                "tags": tags,  # list[str]
                "color_primary": color_primary,  # str
                "text_embed": Vector(textEmbed),  # length-512 vector (list/np.array)
            }
            cursor.execute(
                """
WITH prefilter AS (
  SELECT
    g.id,
    g.image_url,
    g.color_primary,
    g.created_at,
    /* count how many query tags appear in the row's tags */
    (SELECT count(*) FROM unnest(g.tags) t WHERE t = ANY(%(tags)s)) AS tag_hits,
    /* 1 if color matches, else 0 */
    (g.color_primary = %(color_primary)s)::int AS color_match
  FROM garments g
  WHERE g.user_id = %(user_id)s
    AND g.garment_type = %(garment_type)s
),
top20 AS (
  SELECT *
  FROM prefilter
  ORDER BY tag_hits DESC, color_match DESC, created_at DESC
  LIMIT 20
)
SELECT t.image_url
FROM top20 t
JOIN garment_embed e ON e.id = t.id
ORDER BY e.embed <-> %(text_embed)s
LIMIT 5;
""",
                params,
            )
            for row in cursor.fetchall():
                image_url = row[0]
                if image_url not in seen[garment_type]:
                    image = open(image_url, "rb")
                    b64image = base64.b64encode(image.read()).decode("utf-8")
                    image.close()
                    res[garment_type].append(b64image)
                    seen[garment_type].add(image_url)
    return res


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
