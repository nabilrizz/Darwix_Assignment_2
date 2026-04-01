import os
import re
import json
import base64
import logging
import time
from io import BytesIO
from typing import Callable
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, stream_with_context, Response

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

VISUAL_STYLES = {
    "cinematic": {
        "label": "Cinematic Realism",
        "prompt_suffix": "photorealistic, cinematic lighting, 8k, dramatic composition, movie still, sharp focus",
        "negative": "cartoon, illustration, anime, blurry, low quality",
        "icon": "🎬",
    },
    "digital_art": {
        "label": "Digital Art",
        "prompt_suffix": "digital art, vibrant colors, concept art, artstation trending, detailed illustration, professional",
        "negative": "photo, realistic, blurry, low quality, sketch",
        "icon": "🎨",
    },
    "watercolor": {
        "label": "Watercolor Painting",
        "prompt_suffix": "watercolor painting, soft brushstrokes, artistic, warm tones, painted texture, fine art",
        "negative": "digital, photo, harsh lines, dark, gritty",
        "icon": "🖌️",
    },
    "corporate": {
        "label": "Corporate Modern",
        "prompt_suffix": "clean professional photography, business setting, modern office, bright lighting, polished",
        "negative": "dark, gritty, fantasy, cartoon, old fashioned",
        "icon": "💼",
    },
    "comic": {
        "label": "Comic Book",
        "prompt_suffix": "comic book art style, bold lines, vibrant flat colors, panel illustration, graphic novel",
        "negative": "photo, realistic, soft, blurry, watercolor",
        "icon": "💥",
    },
}

SYSTEM_PROMPT_ENGINEER = """You are an expert visual prompt engineer for AI image generation.
Transform a sentence from a business or sales narrative into a richly detailed, visually imaginative image generation prompt.

Rules:
- Make the prompt vivid, specific, and painterly
- Focus on VISUAL elements: composition, lighting, mood, setting, characters, action
- Avoid abstract jargon — translate ideas into concrete visual metaphors
- Keep the prompt to 1–2 sentences (40–70 words)
- Return ONLY the prompt text, no explanations, quotes, or labels"""


def segment_text(text: str) -> list[str]:
    """Segment text into scenes using sentence splitting with smart merging."""
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    merged = []
    buffer = ""
    for i, sent in enumerate(sentences):
        buffer = buffer + " " + sent if buffer else sent
        if len(buffer) > 60 or i == len(sentences) - 1:
            merged.append(buffer)
            buffer = ""

    if len(merged) < 3 and len(sentences) >= 3:
        merged = sentences[:5]

    if len(merged) < 3:
        clauses = re.split(r"(?<=[,;])\s+", text)
        clauses = [c.strip() for c in clauses if len(c.strip()) > 25]
        if len(clauses) >= 3:
            merged = clauses[:6]

    return merged[:6]


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def ollama_available() -> bool:
    try:
        r = requests.get(f"{_ollama_base_url()}/api/tags", timeout=2)
        return r.status_code == 200
    except OSError:
        return False


def engineer_prompt_ollama(
    segment: str,
    scene_index: int,
    total_scenes: int,
    style_key: str,
    narrative_context: str,
) -> str | None:
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
    user_message = f"""Transform this narrative segment into a visual image generation prompt.

Full story (context): {narrative_context}
This segment (scene {scene_index + 1} of {total_scenes}): "{segment}"
Visual style: {VISUAL_STYLES.get(style_key, VISUAL_STYLES["digital_art"])["label"]}

Return only the visual prompt."""

    try:
        r = requests.post(
            f"{_ollama_base_url()}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_ENGINEER},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": 0.75, "num_predict": 220},
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content") or ""
        base = text.strip()
        if not base:
            return None
        style = VISUAL_STYLES.get(style_key, VISUAL_STYLES["digital_art"])
        return f"{base}, {style['prompt_suffix']}"
    except Exception as e:
        logger.warning("Ollama prompt engineering failed: %s", e)
        return None


def _hf_model_id() -> str:
    return os.environ.get(
        "HF_LLM_MODEL",
        "mistralai/Mistral-7B-Instruct-v0.2",
    )


def engineer_prompt_huggingface(
    segment: str,
    scene_index: int,
    total_scenes: int,
    style_key: str,
    narrative_context: str,
) -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return None

    user_block = f"""Transform this narrative segment into a visual image generation prompt.

Full story (context): {narrative_context}
This segment (scene {scene_index + 1} of {total_scenes}): "{segment}"
Visual style: {VISUAL_STYLES.get(style_key, VISUAL_STYLES["digital_art"])["label"]}

Return only the visual prompt."""

    # Mistral Instruct format (matches default HF model)
    prompt = f"<s>[INST] {SYSTEM_PROMPT_ENGINEER}\n\n{user_block} [/INST]"

    url = f"https://api-inference.huggingface.co/models/{_hf_model_id()}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 220,
            "return_full_text": False,
            "temperature": 0.75,
        },
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 503:
            logger.warning("Hugging Face model loading; retry once")
            time.sleep(10)
            r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            base = (data[0].get("generated_text") or "").strip()
        elif isinstance(data, dict) and "generated_text" in data:
            base = (data.get("generated_text") or "").strip()
        else:
            logger.warning("Unexpected HF response shape: %s", type(data))
            return None
        if not base:
            return None
        style = VISUAL_STYLES.get(style_key, VISUAL_STYLES["digital_art"])
        return f"{base}, {style['prompt_suffix']}"
    except Exception as e:
        logger.warning("Hugging Face prompt engineering failed: %s", e)
        return None


def engineer_prompt_heuristic(
    segment: str,
    scene_index: int,
    total_scenes: int,
    style_key: str,
    narrative_context: str,
) -> str:
    """Rule-based visual expansion when no local/cloud LLM is available."""
    style = VISUAL_STYLES.get(style_key, VISUAL_STYLES["digital_art"])
    openings = [
        "Wide cinematic establishing shot of",
        "Dramatic medium shot capturing",
        "Intimate close-up illustrating",
        "Dynamic overhead perspective on",
        "Silhouetted figures against a backdrop of",
        "Sun-drenched modern workspace showing",
    ]
    opener = openings[scene_index % len(openings)]
    ctx = narrative_context.strip()
    if len(ctx) > 200:
        ctx = ctx[:197] + "..."
    core = segment.strip()
    if len(core) > 400:
        core = core[:397] + "..."
    base = (
        f"{opener} a pivotal business moment: {core}. "
        f"Scene {scene_index + 1} of {total_scenes} in a cohesive pitch story; "
        f"narrative arc context: {ctx}"
    )
    return f"{base}, {style['prompt_suffix']}"


def engineer_prompt(
    segment: str,
    scene_index: int,
    total_scenes: int,
    style_key: str,
    narrative_context: str,
) -> tuple[str, str]:
    """
    Returns (final_prompt, source) where source is one of:
    ollama, huggingface, heuristic
    """
    prefer = os.environ.get("LLM_PROVIDER", "auto").lower()

    def try_order() -> list[tuple[str, Callable[[], str | None]]]:
        ollama_fn = lambda: engineer_prompt_ollama(
            segment, scene_index, total_scenes, style_key, narrative_context
        )
        hf_fn = lambda: engineer_prompt_huggingface(
            segment, scene_index, total_scenes, style_key, narrative_context
        )
        if prefer == "ollama":
            return [("ollama", ollama_fn), ("huggingface", hf_fn)]
        if prefer in ("hf", "huggingface"):
            return [("huggingface", hf_fn), ("ollama", ollama_fn)]
        # auto: prefer local Ollama when reachable, else HF, else heuristic
        if ollama_available():
            return [("ollama", ollama_fn), ("huggingface", hf_fn)]
        return [("huggingface", hf_fn), ("ollama", ollama_fn)]

    for name, fn in try_order():
        out = fn()
        if out:
            logger.info("Prompt engineered via %s", name)
            return out, name

    h = engineer_prompt_heuristic(
        segment, scene_index, total_scenes, style_key, narrative_context
    )
    return h, "heuristic"


def generate_image_stability(prompt: str, negative_prompt: str) -> str | None:
    """Generate image using Stability AI API."""
    api_key = os.environ.get("STABILITY_API_KEY")
    if not api_key:
        return None

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "text_prompts": [
            {"text": prompt, "weight": 1},
            {"text": negative_prompt, "weight": -1},
        ],
        "cfg_scale": 7,
        "height": 512,
        "width": 896,
        "steps": 30,
        "samples": 1,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        image_b64 = data["artifacts"][0]["base64"]
        return f"data:image/png;base64,{image_b64}"
    except Exception as e:
        logger.error("Stability AI error: %s", e)
        return None


def _image_to_data_png(image) -> str:
    """Turn PIL Image or raw bytes into a PNG data URL."""
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None and isinstance(image, Image.Image):
        buf = BytesIO()
        image.save(buf, format="PNG")
        raw = buf.getvalue()
    elif isinstance(image, (bytes, bytearray, memoryview)):
        raw = bytes(image)
    else:
        raise TypeError(f"Unexpected image type: {type(image)}")

    b64 = base64.b64encode(raw).decode()
    return f"data:image/png;base64,{b64}"


def generate_image_huggingface_inference(prompt: str, negative_prompt: str) -> str | None:
    """
    Generate an image via Hugging Face Inference (InferenceClient + inference providers).
    The old api-inference.huggingface.co POST is deprecated/unreliable for many models.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return None

    model_id = os.environ.get(
        "HF_IMAGE_MODEL",
        "black-forest-labs/FLUX.1-schnell",
    )

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        logger.warning("Install huggingface_hub and pillow for HF images: pip install huggingface_hub pillow")
        return None

    try:
        client = InferenceClient(token=token)
        kwargs: dict = {"model": model_id}
        if negative_prompt and negative_prompt.strip():
            kwargs["negative_prompt"] = negative_prompt
        try:
            image = client.text_to_image(prompt, **kwargs)
        except Exception as e1:
            logger.warning("HF text_to_image (with options) failed: %s", e1)
            image = client.text_to_image(prompt, model=model_id)
        return _image_to_data_png(image)
    except Exception as e:
        logger.error("Hugging Face InferenceClient text_to_image failed: %s", e)
        return None


def generate_image_pollinations(prompt: str) -> str | None:
    """
    Free, no-key image generation (Pollinations). Used when other backends are unavailable.
    Set POLLINATIONS_FALLBACK=0 to disable.
    """
    flag = os.environ.get("POLLINATIONS_FALLBACK", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return None

    # Keep URL length reasonable; Pollinations truncates internally too
    q = (prompt or "").strip()[:1200]
    if not q:
        return None

    url = f"https://image.pollinations.ai/prompt/{quote(q)}"
    try:
        r = requests.get(
            url,
            timeout=120,
            headers={"User-Agent": "PitchVisualizer/1.0 (educational)"},
        )
        r.raise_for_status()
        raw = r.content
        ct = (r.headers.get("content-type") or "").lower()
        if "image" not in ct and raw[:4] != b"\x89PNG" and raw[:2] != b"\xff\xd8":
            logger.warning("Pollinations returned non-image content-type: %s", ct)
            return None
        b64 = base64.b64encode(raw).decode()
        mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.warning("Pollinations image fallback failed: %s", e)
        return None


def generate_image_dalle(prompt: str) -> str | None:
    """Generate image using OpenAI DALL-E 3."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1792x1024",
        "quality": "standard",
        "response_format": "b64_json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        image_b64 = data["data"][0]["b64_json"]
        return f"data:image/png;base64,{image_b64}"
    except Exception as e:
        logger.error("DALL-E error: %s", e)
        return None


def generate_placeholder_image(prompt: str, index: int, style_key: str) -> str:
    """Generate a stylized SVG placeholder when no image API is available."""
    colors = {
        "cinematic": ["#1a1a2e", "#16213e", "#0f3460", "#e94560"],
        "digital_art": ["#0d0221", "#190335", "#2d1b69", "#11998e"],
        "watercolor": ["#f8e1e7", "#e8d5e8", "#d4e8f0", "#c8e6c9"],
        "corporate": ["#f0f4f8", "#d9e2ec", "#bcccdc", "#9fb3c8"],
        "comic": ["#ff6b35", "#f7c59f", "#efefd0", "#004e89"],
    }

    palette = colors.get(style_key, colors["digital_art"])
    scene_icons = ["🌟", "💡", "🚀", "🎯", "✨", "🔮"]
    icon = scene_icons[index % len(scene_icons)]
    display_text = prompt[:80] + "..." if len(prompt) > 80 else prompt
    words = display_text.split()
    lines = []
    current = []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > 35:
            lines.append(" ".join(current[:-1]))
            current = [word]
    if current:
        lines.append(" ".join(current))

    text_elements = "\n".join(
        [
            f'<text x="448" y="{280 + i * 22}" text-anchor="middle" fill="{palette[-1] if style_key not in ["watercolor", "corporate"] else "#444"}" font-size="12" font-family="Georgia">{line}</text>'
            for i, line in enumerate(lines[:4])
        ]
    )

    bg1, bg2, bg3, accent = palette

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="896" height="512" viewBox="0 0 896 512">
  <defs>
    <linearGradient id="bg{index}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{bg1};stop-opacity:1"/>
      <stop offset="50%" style="stop-color:{bg2};stop-opacity:1"/>
      <stop offset="100%" style="stop-color:{bg3};stop-opacity:1"/>
    </linearGradient>
    <filter id="blur{index}">
      <feGaussianBlur stdDeviation="3"/>
    </filter>
  </defs>
  <rect width="896" height="512" fill="url(#bg{index})"/>
  <circle cx="150" cy="100" r="80" fill="{accent}" opacity="0.15" filter="url(#blur{index})"/>
  <circle cx="750" cy="400" r="120" fill="{accent}" opacity="0.1" filter="url(#blur{index})"/>
  <circle cx="448" cy="256" r="200" fill="{accent}" opacity="0.05"/>
  <text x="448" y="200" text-anchor="middle" font-size="60" font-family="serif">{icon}</text>
  <text x="448" y="240" text-anchor="middle" fill="{accent}" font-size="11" font-family="Georgia" letter-spacing="4" opacity="0.8">SCENE {index + 1} · IMAGE GENERATION PENDING</text>
  {text_elements}
  <text x="448" y="480" text-anchor="middle" fill="white" font-size="10" font-family="monospace" opacity="0.3">Add image API key or HF_TOKEN for real images</text>
</svg>"""

    svg_b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{svg_b64}"


def generate_image(
    prompt: str, negative_prompt: str, index: int, style_key: str
) -> str:
    """Try image backends in order; fall back to Pollinations, then SVG placeholder."""
    result = generate_image_stability(prompt, negative_prompt)
    if result:
        return result
    result = generate_image_huggingface_inference(prompt, negative_prompt)
    if result:
        return result
    result = generate_image_dalle(prompt)
    if result:
        return result
    result = generate_image_pollinations(prompt)
    if result:
        logger.info("Using Pollinations fallback for scene %s", index + 1)
        return result
    logger.warning("No image API available, using placeholder for scene %s", index + 1)
    return generate_placeholder_image(prompt, index, style_key)


@app.route("/")
def index():
    return render_template("index.html", styles=VISUAL_STYLES)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    narrative = data.get("narrative", "").strip()
    style_key = data.get("style", "digital_art")

    if not narrative:
        return jsonify({"error": "No narrative provided"}), 400

    if len(narrative) < 20:
        return jsonify(
            {"error": "Narrative too short. Please provide at least 3 sentences."}
        ), 400

    def event_stream():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing narrative structure...'})}\n\n"
            segments = segment_text(narrative)

            if len(segments) < 3:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Need at least three scenes. Add more sentences or separate ideas with commas/semicolons.'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'segments', 'count': len(segments), 'segments': segments})}\n\n"

            style = VISUAL_STYLES.get(style_key, VISUAL_STYLES["digital_art"])
            panels = []

            for i, segment in enumerate(segments):
                yield f"data: {json.dumps({'type': 'status', 'message': f'Engineering visual prompt for scene {i+1} of {len(segments)}...'})}\n\n"

                engineered_prompt, prompt_source = engineer_prompt(
                    segment, i, len(segments), style_key, narrative
                )

                yield f"data: {json.dumps({'type': 'prompt_ready', 'index': i, 'prompt': engineered_prompt, 'prompt_source': prompt_source})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'message': f'Generating image for scene {i+1}...'})}\n\n"

                image_data = generate_image(
                    engineered_prompt, style.get("negative", ""), i, style_key
                )

                panel = {
                    "index": i,
                    "segment": segment,
                    "engineered_prompt": engineered_prompt,
                    "prompt_source": prompt_source,
                    "image": image_data,
                    "style": style["label"],
                }
                panels.append(panel)

                yield f"data: {json.dumps({'type': 'panel_ready', 'panel': panel})}\n\n"

            yield f"data: {json.dumps({'type': 'complete', 'total': len(panels)})}\n\n"

        except Exception as e:
            logger.error("Generation error: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(event_stream()), content_type="text/event-stream")


@app.route("/health")
def health():
    hf_tok = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    return jsonify(
        {
            "status": "ok",
            "apis": {
                "ollama": ollama_available(),
                "huggingface_llm": hf_tok,
                "huggingface_image": hf_tok,
                "stability_ai": bool(os.environ.get("STABILITY_API_KEY")),
                "openai_dalle": bool(os.environ.get("OPENAI_API_KEY")),
            },
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
