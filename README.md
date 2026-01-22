# LightOnOCR Serverless (RunPod)

## Deploy
1. Build image
   docker build -t yourdockerhub/lighton-ocr .

2. Push image
   docker push yourdockerhub/lighton-ocr

3. Create endpoint
   runpodctl apply -f runpod.yaml

## Input
PDF:
{
  "input": {
    "pdf_base64": "..."
  }
}

Image:
{
  "input": {
    "image_base64": "..."
  }
}

## Notes
- Set HF_TOKEN in RunPod Secrets
- GPU: A100 / L40 / 4090 recommended
- No disk writes
- Serverless auto scales
