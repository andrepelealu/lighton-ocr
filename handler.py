import os
import base64
import io
import torch
import pypdfium2 as pdfium
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import runpod

MODEL_ID = "lightonai/LightOnOCR-2-1B"
PROMPT = "Extract all text from this document page accurately. Preserve layout in markdown."
MAX_SIDE = 1600

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("Loading model...")
model = LightOnOcrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True
).to(DEVICE)

processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")

def prepare_image(image: Image.Image) -> Image.Image:
    if max(image.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(image.size)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.LANCZOS
        )
    return image

def ocr_image(image: Image.Image) -> str:
    image = prepare_image(image)
    inputs = processor(
        images=image,
        text=PROMPT,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )

    return processor.decode(output[0], skip_special_tokens=True)

def handler(job):
    """
    Input:
    {
      "pdf_base64": "..."
    }
    OR
    {
      "image_base64": "..."
    }
    """

    if "pdf_base64" in job["input"]:
        pdf_bytes = base64.b64decode(job["input"]["pdf_base64"])
        pdf = pdfium.PdfDocument(pdf_bytes)

        results = []
        for i, page in enumerate(pdf):
            image = page.render(scale=1.3).to_pil().convert("RGB")
            text = ocr_image(image)
            results.append({"page": i + 1, "text": text})

        return {"pages": results}

    elif "image_base64" in job["input"]:
        image_bytes = base64.b64decode(job["input"]["image_base64"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = ocr_image(image)
        return {"text": text}

    else:
        return {"error": "No valid input provided"}

runpod.serverless.start({"handler": handler})
