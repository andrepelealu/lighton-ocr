import torch
import pypdfium2 as pdfium
from PIL import Image

from transformers.models.lighton_ocr import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor
)

MODEL_ID = "lightonai/LightOnOCR-2-1B"
MAX_SIDE = 1400

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[OCR] Loading model on {DEVICE} ({DTYPE})...")

model = LightOnOcrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True
).to(DEVICE)

processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
model.eval()
print("[OCR] Model loaded")

def _resize(image: Image.Image) -> Image.Image:
    if max(image.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(image.size)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.LANCZOS
        )
    return image

@torch.no_grad()
def ocr_image(image: Image.Image) -> str:
    image = _resize(image)

    inputs = processor(
        images=[image],
        text=["<image>"],   # 🔥 THIS MUST BE EXACTLY THIS
        return_tensors="pt"
    )

    # move to device
    inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # cast only pixel values
    if DEVICE == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()

    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    return processor.decode(output[0], skip_special_tokens=True)



def ocr_pdf_bytes(pdf_bytes: bytes):
    pdf = pdfium.PdfDocument(pdf_bytes)
    results = []

    for i, page in enumerate(pdf):
        print(f"OCR page {i+1}/{len(pdf)}")
        image = page.render(scale=1.2).to_pil().convert("RGB")
        text = ocr_image(image)
        results.append({"page": i + 1, "text": text})

    return results
