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
Your task is to analyze the provided image of a '{garment_type}' and return a single, valid JSON object.

## INSTRUCTIONS
1.  Analyze the image.
2.  Strictly adhere to the output schema and rules below.
3.  Output ONLY the JSON object and nothing else.

## JSON OUTPUT SCHEMA
```json
{{
  "color_primary": "string",
  "material": "string",
  "pattern": "string",
  "tags": ["string"]
}}

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


generate_tags(open("s.webp", "rb").read(), "dress", "webp")
