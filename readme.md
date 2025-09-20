# Garment Recommendation Backend

This project is a Python **Flask** backend server that handles garment uploads, authentication, and hybrid garment recommendation using **CLIP**, **Rembg**, and **Gemini API**. It stores metadata and embeddings in **PostgreSQL** with **pgvector** for similarity search.

---

## Features

* **Authentication**

  * User registration and login with hashed passwords
  * JWT-based authentication and refresh tokens
  * Token rotation and validation

* **Image Upload & Processing**

  * Accepts base64-encoded images
  * Removes background with [rembg](https://github.com/danielgatis/rembg)
  * Generates embeddings with [OpenAI CLIP](https://github.com/openai/CLIP)
  * Extracts garment features (color, material, pattern, tags) via Gemini API

* **Database**

  * Garment metadata stored in `garments`
  * Image embeddings stored in `garment_embed (vector(512))`
  * User and refresh token tables

* **Hybrid Retrieval**

  * Rank by tag overlap → color match → vector similarity
  * Returns top 5 matching garments per query

* **API Endpoints**

  * `POST /register` – Register a new user
  * `POST /login` – Login and get JWT + refresh token
  * `POST /updatejwt` – Refresh JWT using refresh token
  * `POST /pushdb` – Upload a garment image
  * `POST /chat` – Experimental conversational garment search

---

## Project Structure

```
project/
│── app.py                # Flask backend
│── geminiAPI.py           # Gemini API integration
│── garments/              # Uploaded images stored here
│── .env                   # Environment variables
```

---

## Environment Variables

Set these in `.env`:

```
user=postgres_user
pass=postgres_password
host=localhost
port=5432
JWTSECRET=your_jwt_secret
```

---

## Database Schema

### users

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    email TEXT UNIQUE NOT NULL,
    hashed_pass TEXT NOT NULL
);
```

### refresh\_tokens

```sql
CREATE TABLE refresh_tokens (
    user_id UUID PRIMARY KEY REFERENCES users(id),
    token_hash TEXT NOT NULL,
    exp TIMESTAMP WITH TIME ZONE NOT NULL
);
```

### garments

```sql
CREATE TABLE garments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID NOT NULL REFERENCES users(id),
    garment_type TEXT,
    image_url TEXT NOT NULL,
    color_primary TEXT,
    material TEXT,
    pattern TEXT,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

### garment\_embed

```sql
CREATE TABLE garment_embed (
    id UUID PRIMARY KEY REFERENCES garments(id),
    embed VECTOR(512) NOT NULL
);
```

---

## API Reference

### `POST /register`

Register a new user.

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

Returns: JWT + refresh token.

### `POST /login`

Login and get tokens.

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

### `POST /pushdb`

Upload a garment.

```json
{
  "jwt": "access_jwt",
  "img": "base64_encoded_image",
  "type": "top"
}
```

### `POST /updatejwt`

Refresh JWT.

```json
{
  "jwt": "old_jwt",
  "refreshToken": "refresh_token"
}
```

### `POST /chat`

Experimental conversation → garment results.

---

## Setup

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run Flask server:

   ```bash
   python app.py
   ```
4. Visit: `http://0.0.0.0:5000`

---

## Tech Stack

* Flask (Python)
* PostgreSQL + pgvector
* Rembg (background removal)
* CLIP ViT-B/32 (embeddings)
* JWT (auth)
* Gemini API (feature extraction)

---

## Notes

* Images saved under `garments/{user_id}/{garment_id}.jpg`
* JWT expires in **24 hours**
* Refresh token expires in **90 days**

