#!/usr/bin/env python3
# Terminal chat with Gemini 1.5 Flash using google-genai
# pip install google-genai python-dotenv
# Usage: python chat_gemini.py [--model gemini-1.5-flash-latest] [--system "you are ..."]

import os
import sys
from flask import json
import argparse
from typing import List, Optional, Tuple
from dotenv import load_dotenv

from google import genai
from google.genai import types

BANNER = """\
Gemini Chat (google-genai)
Commands:
  /image <path>     attach an image to your next message
  /clear            clear conversation history
  /save <path>      save transcript to a text file
  /exit             quit
"""

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
- When you have enough info, your final output **must be a single JSON object** that groups garments by type.
- The keys of this object are the garment types (e.g., "top", "bottom", "jewelry").
- The value for each key must be a **list** of garment objects.

Schema for each individual garment JSON:
{"{"}{
     JSON_SCHEMA + '   "image_description": "<max 6-word descriptive phrase>"'
}
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
- Include jewelry suggestions as separate JSONs, grouped under the "jewelry" key in the final output.

Final formatting:
- When you’re asking a question, output ONLY the question (no JSON).
- When you’re delivering results, output ONLY JSON. The structure must be an object with garment types as keys and lists of garments as values, like this example:
```json
{{
  "top": [
    {{
      "garment_type": "top",
      "color_primary": "white",
      "material": "cotton",
      "pattern": "solid",
      "tags": ["fitted", "crewneck", "short-sleeve", "minimal", "casual"],
      "image_description": "white fitted crewneck cotton tee"
    }}
  ],
  "bottom": [
    {{
      "garment_type": "bottom",
      "color_primary": "blue",
      "material": "denim",
      "pattern": "solid",
      "tags": ["straight", "high-waist", "denim", "casual", "streetwear"],
      "image_description": "high-waist straight leg blue jeans"
    }}
  ],
  "jewelry": [
    {{
      "garment_type": "jewelry",
      "color_primary": "gold",
      "material": "metal",
      "pattern": "solid",
      "tags": ["necklace", "delicate", "gold", "minimal", "layered"],
      "image_description": "delicate layered gold necklace"
    }}
  ]
}}"""
TYPES_CONVO_PROMPT = types.Part.from_text(text=CONVO_PROMPT)

def read_image_bytes(path: str) -> Tuple[bytes, str]:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    # Basic mapping; expand as needed
    mime = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "webp": "webp",
        "gif": "gif",
        "bmp": "bmp",
        "tiff": "tiff",
        "tif": "tiff",
        "heic": "heic",
        "heif": "heif",
    }.get(ext, None)
    if mime is None:
        raise ValueError(f"Unsupported image format: .{ext}")
    with open(path, "rb") as f:
        return f.read(), mime

def build_contents(
    system_prompt: Optional[str],
    turns: List[Tuple[List[types.Part], str]]
) -> List[types.Content]:
    """
    Build a list of types.Content for the conversation.

    turns: list of (user_parts, assistant_text)
           - user_parts: list[types.Part] (text and/or image bytes)
           - assistant_text: str (model reply) or "" for no reply yet
    """
    contents: List[types.Content] = []

    # Optional system message: send as a model-preface instruction.
    # Some SDKs have dedicated fields; here we include as an initial user role instruction.
    if system_prompt:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_prompt)]
            )
        )

    for user_parts, assistant_text in turns:
        contents.append(types.Content(role="user", parts=user_parts))
        if assistant_text:
            contents.append(
                types.Content(role="model", parts=[types.Part.from_text(text=assistant_text)])
            )

    return contents

def main():
    parser = argparse.ArgumentParser(description="Terminal chat for Gemini via google-genai")
    parser.add_argument("--model", default="gemma-3-12b-it", help="Model name")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GOOGLEKEY")
    if not api_key:
        raise ValueError("GOOGLEKEY environment variable not set!")

    client = genai.Client(api_key=api_key)

    model_name = args.model
    system_prompt = CONVO_PROMPT
    print(BANNER)
    if system_prompt:
        print(f"[system] {system_prompt}\n")

    # Each turn is (user_parts, assistant_text)
    turns: List[Tuple[List[types.Part], str]] = []
    pending_images: List[Tuple[bytes, str]] = []  # list of (data, mime_suffix), e.g., ("jpeg")

    def attach_pending_images(parts: List[types.Part]):
        for data, fmt in pending_images:
            parts.append(types.Part.from_bytes(data=data, mime_type=f"image/{fmt}"))
        pending_images.clear()

    try:
        while True:
            try:
                user_line = input("you › ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_line:
                continue

            # Handle commands
            if user_line.startswith("/"):
                pieces = user_line.split(maxsplit=1)
                cmd = pieces[0].lower()
                arg = pieces[1] if len(pieces) == 2 else ""

                if cmd == "/exit":
                    print("Bye!")
                    break

                elif cmd == "/clear":
                    turns.clear()
                    pending_images.clear()
                    print("(history cleared)")

                elif cmd == "/image":
                    if not arg:
                        print("Usage: /image <path>")
                        continue
                    try:
                        data, fmt = read_image_bytes(arg)
                        pending_images.append((data, fmt))
                        print(f"(attached image: {arg})")
                    except Exception as e:
                        print(f"Error attaching image: {e}")

                elif cmd == "/save":
                    if not arg:
                        print("Usage: /save <path>")
                        continue
                    try:
                        with open(arg, "w", encoding="utf-8") as f:
                            if system_prompt:
                                f.write(f"[SYSTEM]\n{system_prompt}\n\n")
                            for i, (user_parts, assistant) in enumerate(turns, 1):
                                # Extract text parts
                                user_texts = [p.text for p in user_parts if hasattr(p, "text")]
                                f.write(f"[USER {i}]\n")
                                for ut in user_texts:
                                    f.write(ut + "\n")
                                image_count = sum(1 for p in user_parts if not hasattr(p, "text"))
                                if image_count:
                                    f.write(f"(+{image_count} image(s))\n")
                                f.write("\n[ASSISTANT]\n")
                                f.write((assistant or "") + "\n\n")
                        print(f"(saved to {arg})")
                    except Exception as e:
                        print(f"Error saving transcript: {e}")

                else:
                    print("Unknown command. Available: /image, /clear, /save, /exit")
                continue

            # Build user parts for this turn
            user_parts: List[types.Part] = [types.Part.from_text(text=user_line)]
            attach_pending_images(user_parts)

            # Build full contents (history + this user turn)
            contents = build_contents(system_prompt, turns + [(user_parts, "")])

            try:
                # Simple, non-streaming call mirroring your usage
                cfg = types.GenerateContentConfig()
                resp_text = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=cfg,
                ).text

                resp_text = resp_text or ""  # guard None
            except Exception as e:
                resp_text = f"(error from API: {e})"

            # Record the turn
            turns.append((user_parts, resp_text))

            # Print assistant
            print("bot › " + resp_text + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")

if __name__ == "__main__":
    main()
