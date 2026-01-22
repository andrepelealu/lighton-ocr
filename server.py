from fastapi import FastAPI, UploadFile, File
from ocr import ocr_image, ocr_pdf_bytes

app = FastAPI(title="LightOn OCR API")


@app.post("/ocr/image")
async def ocr_from_image(file: UploadFile = File(None), url: str = None):
    if not file and not url:
        return {"error": "file or url is required"}

    if file:
        text = ocr_image(file.file)
    else:
        text = ocr_image(url=url)

    return {"text": text}


@app.post("/ocr/pdf")
async def ocr_from_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    pages = ocr_pdf_bytes(pdf_bytes)
    return {"pages": pages}
