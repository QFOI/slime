"""Lossless image translation and HF preprocessing for agent training turns."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image


def image_content(value):
    """Canonical HF content; preserve image bytes, fail on unsupported modalities.

    Only inline images are accepted: a training worker must not read arbitrary
    agent-selected local paths or fetch a changing remote URL during replay.
    Text-only content stays a string for existing tool chat templates.
    """
    if value is None or isinstance(value, str):
        return value or ""
    blocks = value if isinstance(value, list) else [value]
    result = []
    for block in blocks:
        if isinstance(block, str):
            result.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            raise ValueError("Unsupported robot model content block")
        kind = block.get("type")
        if kind in {"text", "input_text", "output_text"}:
            result.append({"type": "text", "text": block.get("text", "")})
        elif kind in {"image", "image_url", "input_image"}:
            source = block.get("source") or {}
            if source.get("type") == "base64":
                uri = f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
            elif kind == "image" and block.get("data"):
                uri = f"data:{block.get('mimeType', 'image/png')};base64,{block['data']}"
            else:
                uri = block.get("image_url") or block.get("image") or source.get("url")
                if isinstance(uri, dict):
                    uri = uri.get("url")
            if not isinstance(uri, str) or not uri.startswith("data:image/") or ";base64," not in uri:
                raise ValueError("Robot training requires inline base64 image data")
            result.append({"type": "image", "image": uri})
        else:
            raise ValueError(f"Unsupported robot training content type: {kind}")
    if all(block["type"] == "text" for block in result):
        return "\n".join(block["text"] for block in result)
    return result


@dataclass
class EncodedPrompt:
    text: str
    token_ids: list[int]
    image_data: list[str]
    train_inputs: dict | None


def encode_prompt(messages, tools, processor, template_kwargs) -> EncodedPrompt:
    """Use the same checkpoint's processor and preserve image order across turns."""
    images, image_data = [], []
    for message in messages:
        for block in message["content"] if isinstance(message.get("content"), list) else []:
            if block["type"] == "image":
                uri = block["image"]
                raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    images.append(image.convert("RGB"))
                image_data.append(uri)
    text = processor.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    output = processor(text=[text], images=images or None, return_tensors="pt", padding=False)
    ids = output["input_ids"][0].tolist()
    train_inputs = {
        key: value.cpu() if hasattr(value, "cpu") else value
        for key, value in output.items()
        if key not in {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
    }
    if images and not train_inputs:
        raise ValueError("Processor discarded robot images; a vision-language checkpoint is required")
    return EncodedPrompt(text, ids, image_data, train_inputs or None)
