import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from PIL import Image
import requests
from io import BytesIO
import pypdfium2 as pdfium

MODEL_ID = "lightonai/LightOnOCR-2-1B"

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


def ocr_pil_image(image: Image.Image) -> str:
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

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, skip_special_tokens=True)


def ocr_image(file=None, url=None) -> str:
    if url:
        r = requests.get(url)
        r.raise_for_status()
        image = Image.open(BytesIO(r.content)).convert("RGB")
    else:
        image = Image.open(file).convert("RGB")

    return ocr_pil_image(image)


def ocr_pdf_bytes(pdf_bytes: bytes):
    pdf = pdfium.PdfDocument(pdf_bytes)

    results = []

    for i in range(len(pdf)):
        page = pdf[i]
        pil_image = page.render_to_pil(
            scale=2  # IMPORTANT for OCR quality
        )
        print(f"OCR page {i+1}/{len(pdf)}")
        text = ocr_pil_image(pil_image)

        results.append({
            "page": i + 1,
            "text": text
        })

    return results


    return results
