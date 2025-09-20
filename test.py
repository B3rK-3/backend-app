import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector
import os
from dotenv import load_dotenv
import torch
import clip
from collections import defaultdict

load_dotenv()

conn = psycopg2.connect(
    database="postgres",
    user=os.environ["user"],
    password=os.environ["pass"],
    host=os.environ["host"],
    port=os.environ["port"],
)
register_vector(conn)   
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

cursor = conn.cursor()


def clipText(text: str):
    with torch.no_grad():
        textTokens = clip.tokenize(text).to(device)
        text_embedding = model.encode_text(textTokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.cpu().tolist()[0]


def get_conn():
    return psycopg2.connect(
        database="postgres",
        user=os.environ["user"],
        password=os.environ["pass"],
        host=os.environ["host"],
        port=os.environ["port"],
    )


def getGarments(jsonObj: dict, userID: str):
    """
    Parses the database and return images for the given garments
    """
    conn = get_conn()
    cursor = conn.cursor()

    res = defaultdict(set)

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
            res[garment_type].update(set(cursor.fetchall()))
    return res

print(
    getGarments(
        {
            "top": [
                {
                    "garment_type": "top",
                    "color_primary": "white",
                    "material": "cotton",
                    "pattern": "solid",
                    "tags": [
                        "collared",
                        "long-sleeve",
                        "slim",
                        "fitted",
                        "casual",
                        "minimal",
                    ],
                    "image_description": "slim white cotton shirt",
                }
            ],
            "bottom": [
                {
                    "garment_type": "bottom",
                    "color_primary": "navy",
                    "material": "wool",
                    "pattern": "solid",
                    "tags": ["tailored", "casual"],
                    "image_description": "navy wool trousers",
                }
            ],
            "jewelry": [
                {
                    "garment_type": "jewelry",
                    "color_primary": "gold",
                    "material": "metal",
                    "pattern": "solid",
                    "tags": ["ring", "minimal", "gold"],
                    "image_description": "gold minimal ring",
                },
                {
                    "garment_type": "jewelry",
                    "color_primary": "gold",
                    "material": "metal",
                    "pattern": "solid",
                    "tags": ["bracelet", "minimal", "gold"],
                    "image_description": "gold minimal bracelet",
                },
            ],
        },
        "019964db-ee31-7676-8ba1-0ab63d9c2ceb",
    )
)
