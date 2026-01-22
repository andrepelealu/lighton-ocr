import os
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = ""

import torch
torch._dynamo.disable()

from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import pypdfium2 as pdfium

DEVICE = "cuda"
MODEL_ID = "lightonai/lighton-ocr"

processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
model = LightOnOcrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda"
)
model.eval()
model.config.use_cache = False


def _resize(img: Image.Image, max_side=1600):
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    return img


@torch.no_grad()
def ocr_image(image: Image.Image) -> str:
    image = _resize(image)

    inputs = processor(
        images=[image],
        text=["<image>"],   # MUST be exactly this
        return_tensors="pt"
    )

    inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].half()

    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    return processor.decode(output[0], skip_special_tokens=True)


def ocr_pdf_bytes(pdf_bytes: bytes):
    pdf = pdfium.PdfDocument(pdf_bytes)
    pages = []

    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render_to(
            pdfium.BitmapConv.pil_image,
            scale=2
        )
        text = ocr_image(pil)
        pages.append(text)

    return pages
