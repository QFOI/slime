"""Small explicit protocol fixtures; these are not a production tokenizer/VLM."""

import copy
import json

import numpy as np
import torch


class ByteTokenizer:
    def encode(self, text, **kwargs):
        return [b + 1 for b in text.encode()]

    def decode(self, ids, **kwargs):
        return bytes(int(i) - 1 for i in ids).decode()


class PixelProcessor:
    def __init__(self):
        self.tokenizer = ByteTokenizer()
        self.image_processor = self

    def apply_chat_template(self, messages, tools=None, **kwargs):
        messages = copy.deepcopy(messages)
        for message in messages:
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "image":
                        item["image"] = "<image>"
        return json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)

    def __call__(self, *, text, images=None, **kwargs):
        result = {"input_ids": torch.tensor([self.tokenizer.encode(text[0])])}
        if images:
            result["pixel_values"] = torch.tensor(
                np.stack([np.asarray(i.resize((8, 8)), dtype=np.float32).transpose(2, 0, 1) / 255 for i in images])
            )
        return result
