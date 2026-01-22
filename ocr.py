from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
import torch, os

MODEL_ID = "lightonai/LightOnOCR-2-1B"

# --- torch stability ---
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.disable()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# --- MODEL ---
model = LightOnOcrForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

# --- PROCESSOR (IMPORTANT FIX) ---
processor = LightOnOcrProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False,
)
