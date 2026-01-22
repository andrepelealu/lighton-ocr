import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from PIL import Image
import requests
from io import BytesIO

MODEL_ID = "lightonai/LightOnOCR-2-1B"

# device & dtype (EXACTLY like your working code)
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
dtype = torch.float32 if device == "mps" else torch.bfloat16

print("Device:", device, "dtype:", dtype)

model = LightOnOcrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
).to(device)

processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)


def load_image(file=None, url=None):
    if url:
        r = requests.get(url)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    else:
        return Image.open(file).convert("RGB")


@torch.inference_mode()
def ocr_image(file=None, url=None) -> str:
    image = load_image(file=file, url=url)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]

    return processor.decode(generated_ids, skip_special_tokens=True)
