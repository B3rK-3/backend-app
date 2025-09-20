# To run this code you need to install the following dependencies:
# pip install google-genai

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json


load_dotenv()

# It's good practice to handle the case where the key might be missing
api_key = os.environ.get("GOOGLEKEY")
if not api_key:
    raise ValueError("GOOGLEKEY environment variable not set!")

client = genai.Client(api_key=api_key)

# The model name in your code was for a non-existent model.
# Using a valid Gemini model with vision capabilities.
MODEL_NAME = "gemini-1.5-flash-latest"
ALLOWED_TAGS = {
    "top": [
        "oversized",
        "slim",
        "cropped",
        "boxy",
        "fitted",
        "vneck",
        "crewneck",
        "scoop",
        "collared",
        "turtleneck",
        "short-sleeve",
        "long-sleeve",
        "sleeveless",
        "puff-sleeve",
        "casual",
        "formal",
        "sporty",
        "graphic",
        "minimal",
        "hoodie",
    ],
    "bottom": [
        "relaxed",
        "skinny",
        "wide-leg",
        "straight",
        "tailored",
        "cargo",
        "joggers",
        "pleated",
        "wrap",
        "shorts",
        "bermuda",
        "mid-thigh",
        "knee-length",
        "cropped",
        "mini",
        "midi",
        "maxi",
        "high-waist",
        "low-rise",
        "elastic",
        "drawstring",
        "denim",
        "chino",
        "athletic",
        "beach",
        "casual",
        "sporty",
        "streetwear",
        "summer",
    ],
    "dress": [
        "a-line",
        "sheath",
        "bodycon",
        "fit-and-flare",
        "wrap",
        "mini",
        "midi",
        "maxi",
        "sleeveless",
        "halter",
        "off-shoulder",
        "strapless",
        "evening",
        "casual",
        "cocktail",
        "boho",
        "elegant",
        "summer",
        "vneck",
    ],
    "jewelry": [
        "necklace",
        "ring",
        "bracelet",
        "earring",
        "minimal",
        "statement",
        "layered",
        "chunky",
        "delicate",
        "gold",
        "silver",
        "platinum",
        "pearl",
        "gemstone",
        "leather",
        "geometric",
        "floral",
        "heart",
        "star",
        "religious",
        "vintage",
    ],
    "hat": [
        "baseball",
        "beanie",
        "bucket",
        "fedora",
        "snapback",
        "visor",
        "wide-brim",
        "casual",
        "sporty",
        "formal",
        "retro",
        "minimal",
    ],
}

JSON_SCHEMA = """
  "garment_type": "top" | "bottom" | "dress" | "jewelry" | "hat",
  "color_primary": "<lowercase one word color, e.g. 'black','white','navy','gold'>",
  "material": "cotton" | "linen" | "denim" | "leather" | "metal" | "wool" | "satin" | "silk">",
  "pattern": "solid" | "striped" | "plaid" | "floral" | "graphic" | "textured" | "other",
  "tags": [5..10 strings drawn ONLY from the allowed list for that garment_type],
"""

CONVO_PROMPT = f"""
You are a structured fashion planning assistant. 
You DO NOT analyze images. You ask concise follow-up questions, then generate JSON describing garments.

Inputs available:
- gender: "male" | "female" | "unspecified"
- garment_type: optional; if the user implies or states a garment (e.g., “red silk dress”), treat it as the primary garment_type.
- User free-text preferences and answers to your questions.

Your goals:
1) Ask the minimum clarifying questions to confidently produce outputs (occasion, vibe, setting, season, formality).
2) Generate valid JSON objects for:
   a) The primary garment (if implied or requested by the user).
   b) Up to 4 jewelry pieces (each separate JSON) with garment_type one of: "necklace","ring","bracelet","earring".
3) Respect gender constraints:
   - If gender = "male", do NOT propose dress JSON (unless user explicitly asks for a dress), include top AND bottom.
   - If gender = "female" or "unspecified", dresses are allowed if context implies them.

Output rules:
- Output JSON only (no extra text). If you need to ask a question, ask it plainly (no JSON) and WAIT for the answer.
- When you have enough info, return:
  - A list of multiple style options (e.g., {{ "top": [ {{...}}, {{...}} ], "jewelry": [{{...}} ] }}).

Schema for each garment JSON:
{"{"}{JSON_SCHEMA + '  "image_description": "<max 6-word descriptive phrase>"'}
{"}"}

ALLOWED_TAGS = {ALLOWED_TAGS}


Hard constraints:
- Tags MUST be a subset of ALLOWED_TAGS[garment_type].
- Use lowercase for color/material/pattern.
- If a user request conflicts with allowed tags, keep attributes as requested but choose the CLOSEST valid tags; do not invent tags.
- Jewelry: generate up to 4 separate JSONs (necklace, ring, bracelet, earring) when asked or when the user implies a full look. Each jewelry JSON must have garment_type="jewelry" and include one of ["necklace","ring","bracelet","earring"] in tags (and possibly "gold","delicate", etc.).


Questioning strategy:
- Ask at most 1–2 short follow-ups if needed: occasion, setting/location, formality level, season, preferred vibe (elegant, casual, sporty, boho, minimal).
- If the user already provided enough info, stop asking and produce JSON.

Multiple style options:
- For vague prompts (e.g., “going on a date”), produce up to 2–3 distinct primary options aligned to gender and setting (e.g., cafe: smart-casual, cozy-minimal).
- Include jewelry suggestions as separate JSONs.

Final formatting:
- When you’re asking a question, output ONLY the question (no JSON).
- When you’re delivering results, output ONLY JSON (no prose).

"""
TYPES_CONVO_PROMPT = types.Part.from_text(text=CONVO_PROMPT)

print(CONVO_PROMPT)


def generate_tags(image_bytes: bytes, garment_type: str, image_format: str) -> dict:
    """
    Generates fashion tags for an image using a Gemini model.

    Args:
        image_bytes: The image data in bytes.
        garment_type: The type of garment (e.g., "dress", "top").
        image_format: The mime type of the image (e.g., "webp", "jpeg").
    """
    if garment_type not in ALLOWED_TAGS:
        raise ValueError(
            f"Invalid garment_type: {garment_type}. Must be one of {list(ALLOWED_TAGS.keys())}"
        )

    # Dynamically select the correct list of tags
    allowed_list = ALLOWED_TAGS[garment_type]

    # Use an f-string for a much clearer, more structured prompt
    prompt_text = f"""
You are an expert fashion tagging API.
Your task is to analyze the provided image of a '{
        garment_type
    }' and return a single, valid JSON object.

## INSTRUCTIONS
1.  Analyze the image.
2.  Strictly adhere to the output schema and rules below.
3.  Output ONLY the JSON object and nothing else.

## JSON OUTPUT SCHEMA
```json
Schema for each garment JSON:
{"{"}{JSON_SCHEMA + '  "image_description": "<max 6-word descriptive phrase>"'}
{"}"}

## ALLOWED TAGS
{json.dumps(allowed_list)}
"""

    CONTENT = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
                types.Part.from_bytes(
                    data=image_bytes, mime_type=f"image/{image_format}"
                ),
            ],
        )
    ]
    generate_content_config = types.GenerateContentConfig()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=CONTENT,
        config=generate_content_config,
    ).text

    if response[:7] != "```json":
        return None
    else:
        return json.loads(response[8:-4])


def getConvoResponse(convo: list):
    """
    takes in convo and streams the text response

    Parameters:
        convo (str): list of past messages
    Returns:
        res (str | dict): ai response or dict with shape { "top": [ {...}, {...} ], "jewelry": [ {...} ] }
    """
    history = build(convo)
    cfg = types.GenerateContentConfig()
    print(convo, history)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=history,
        config=cfg,
    ).text

    if response[:7] != "```json":
        return response
    else:
        return json.loads(response[8:-4])


def build(convo):
    """Build ai history from convo"""
    contents: list[types.Content] = []

    contents.append(types.Content(role="user", parts=[TYPES_CONVO_PROMPT]))
    # {"convo": [{"content": "Hi", "role": "user"}, {"content": "Hmm, I received an unexpected response format.", "role": "model"}, {"content": "H", "role": "user"}]F
    for message in convo:
        role = message["role"]
        content = message["content"]
        contents.append(types.Content(role=role, parts=[types.Part.from_text(content)]))
    return contents
