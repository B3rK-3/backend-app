import psycopg2
from pgvector.psycopg2 import register_vector
from pgvector import Vector
import os
from dotenv import load_dotenv
import torch
import clip
from collections import defaultdict
import base64

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
                    # image = open(image_url, "rb")
                    # b64image = base64.b64encode(image.read()).decode("utf-8")
                    # image.close()
                    res[garment_type].append(image_url)
                    seen[garment_type].add(image_url)
    return res


userID = "01996955-cdbd-7c57-88e5-0af497f8e790"
print(
    getGarments(
        {
            
            "bottom": [
                {
                    "garment_type": "bottom",
                    "color_primary": "beige",
                    "material": "wool",
                    "pattern": "solid",
                    "tags": ["tailored", "straight", "high-waist", "formal", "classic"],
                    "image_description": "trousers for old money style",
                }
            ]
        },
        userID,
    )
)

conn = get_conn()
cursor = conn.cursor()
cursor.execute("""SELECT
  column_name,
  data_type,
  character_maximum_length,
  is_nullable,
  column_default
FROM
  information_schema.columns
WHERE
  table_name = 'garment_embed';""")
print(cursor.fetchall())