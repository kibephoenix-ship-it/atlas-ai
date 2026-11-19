import base64
import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI
# Update line 8 to include FileResponse:
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
client = genai.Client()

HTML_PATH = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
def home():
  if HTML_PATH.exists():
    return HTML_PATH.read_text(encoding="utf-8")
  return "<h1>Error: index.html not found</h1>"


@app.get("/config")
def get_config():
  return JSONResponse({"google_client_id": os.getenv("GOOGLE_CLIENT_ID", "")})


@app.get("/favicon.ico", response_class=FileResponse)
def get_favicon():
  return FileResponse(Path(__file__).parent / "image.png")


class MessageItem(BaseModel):
  role: str
  text: str
  persona: Optional[str] = "decision"
  file_name: Optional[str] = None
  mime_type: Optional[str] = None
  file_data: Optional[str] = None


class ChatRequest(BaseModel):
  history: list[MessageItem]


@app.post("/chat")
def chat(req: ChatRequest):
  def generate():
    try:
      last_msg = req.history[-1] if req.history else None
      active_persona = last_msg.persona if last_msg else "decision"

      # Nano Banana Imagen 3 Generation
      if active_persona == "banana" and last_msg:
        yield f"data: {json.dumps({'text': 'Generating image with Imagen 3...\\n\\n'})}\n\n"
        result = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=last_msg.text,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio="16:9",
            ),
        )
        for generated_image in result.generated_images:
          img_b64 = base64.b64encode(generated_image.image.image_bytes).decode(
              "utf-8"
          )
          yield f"data: {json.dumps({'image': img_b64, 'text': f'Generated image for: **{last_msg.text}**'})}\n\n"
        return

      # Standard Streaming Text + Search Grounding
      persona_instructions = {
          "decision": (
              "You are Atlas Pro, an elite strategic decision-making assistant."
              " Search the live web for factual data, weigh trade-offs"
              " rigorously, and conclude with a decisive recommendation"
              " formatted in clean Markdown."
          ),
          "code": (
              "You are Atlas Pro, a Principal Software Architect. Write clean,"
              " production-ready code. When asked to build web UIs, provide"
              " complete, self-contained HTML/CSS/JS inside a single ```html"
              " code block so it renders cleanly in the Live Preview panel."
          ),
          "research": (
              "You are Atlas Pro, an expert Research Analyst. Perform thorough"
              " live Google Search grounding, cite data points accurately, and"
              " output structured tables and synthesis."
          ),
          "writer": (
              "You are Atlas Pro, a professional technical and creative writer."
              " Craft articulate, engaging, and polished prose with precise"
              " vocabulary and clean formatting."
          ),
      }
      system_prompt = persona_instructions.get(
          active_persona, persona_instructions["decision"]
      )

      contents = []
      for msg in req.history:
        parts = []
        if msg.file_data and msg.mime_type:
          raw_bytes = base64.b64decode(msg.file_data)
          parts.append(
              types.Part.from_bytes(data=raw_bytes, mime_type=msg.mime_type)
          )
        parts.append(types.Part.from_text(text=msg.text))
        contents.append(types.Content(role=msg.role, parts=parts))

      response = client.models.generate_content_stream(
          model="gemini-3.5-flash-lite",
          contents=contents,
          config=types.GenerateContentConfig(
              system_instruction=system_prompt,
              tools=[{"google_search": {}}],
          ),
      )
      for chunk in response:
        if chunk.text:
          yield f"data: {json.dumps({'text': chunk.text})}\n\n"

    except Exception as e:
      yield f"data: {json.dumps({'text': f'⚠️ Error: {str(e)}'})}\n\n"

  return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
  import uvicorn

  port = int(os.environ.get("PORT", 8000))
  uvicorn.run("app:app", host="0.0.0.0", port=port)