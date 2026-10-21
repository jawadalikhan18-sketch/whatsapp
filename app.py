import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from google import genai

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("whatsapp-ai-bot")

# Railway Variables (set these in Railway > your service > Variables)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v23.0")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

app = FastAPI(title="WhatsApp AI Bot", version="1.0.0")


def ask_gemini(prompt: str) -> str:
    """Generate an answer using the Gemini API."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    answer = (response.text or "").strip()
    if not answer:
        return "Sorry, I couldn't generate a response. Please try again."
    return answer[:4000]


def send_whatsapp_message(to: str, message: str) -> None:
    """Send a text message through the WhatsApp Cloud API."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        raise RuntimeError("WHATSAPP_TOKEN or PHONE_NUMBER_ID is not configured")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": message[:4000]},
    }

    response = requests.post(url, json=payload, headers=headers, timeout=20)
    if not response.ok:
        logger.error("WhatsApp API error %s: %s", response.status_code, response.text[:1000])
        response.raise_for_status()


@app.get("/")
async def home():
    return {
        "status": "online",
        "service": "WhatsApp AI Bot",
        "webhook": "/webhook",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta calls this endpoint to verify the webhook subscription."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token and token == VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge, status_code=200)

    return JSONResponse({"error": "verification failed"}, status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    """Receive Meta webhook events and reply to incoming text messages."""
    try:
        data: dict[str, Any] = await request.json()

        # Ignore status callbacks and non-WhatsApp events without failing.
        entries = data.get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                for message in messages:
                    if message.get("type") != "text":
                        logger.info("Ignoring non-text WhatsApp message.")
                        continue

                    sender = message.get("from")
                    user_text = message.get("text", {}).get("body", "").strip()
                    if not sender or not user_text:
                        continue

                    logger.info("Received a text message from a WhatsApp user.")
                    try:
                        ai_reply = ask_gemini(user_text)
                        send_whatsapp_message(sender, ai_reply)
                    except Exception:
                        logger.exception("Failed to generate or send a WhatsApp reply.")
                        # Return 200 so Meta does not repeatedly retry this event.
        return {"status": "ok"}

    except Exception:
        logger.exception("Webhook payload processing failed.")
        # Return 200 for malformed/unexpected callbacks to avoid endless retries.
        return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
