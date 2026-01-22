from fastapi import FastAPI, UploadFile, File
from ocr import ocr_image
import tempfile

app = FastAPI(title="LightOn OCR API")


@app.post("/ocr/image")
async def ocr_from_image(file: UploadFile = File(None), url: str = None):
    if not file and not url:
        return {"error": "file or url is required"}

    if file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            text = ocr_image(file=tmp.name)
    else:
        text = ocr_image(url=url)

    return {"text": text}
