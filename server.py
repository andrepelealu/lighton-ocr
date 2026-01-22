import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ocr import ocr_image, ocr_pdf_bytes

app = FastAPI(title="LightOnOCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ocr/image")
async def ocr_image_api(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    text = ocr_image(image)
    return {"text": text}

@app.post("/ocr/pdf")
async def ocr_pdf_api(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    pages = ocr_pdf_bytes(pdf_bytes)
    return {
        "total_pages": len(pages),
        "pages": pages
    }
