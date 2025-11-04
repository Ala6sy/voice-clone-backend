# FILE: main.py
# -------------------------------------------
# FastAPI Backend for Voice Cloning Lab
# -------------------------------------------
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from pathlib import Path
from pydub import AudioSegment
from TTS.api import TTS

# إنشاء التطبيق
app = FastAPI(title="Voice Cloning Lab API")

# إعداد CORS للسماح بالوصول من الواجهة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مجلد التخزين
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ذاكرة مؤقتة للأصوات المسجّلة
voices = {}

# تحميل نموذج Coqui XTTS (يدعم العربية)
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)


@app.get("/")
def root():
    return {"status": "ok", "message": "Voice Cloning Backend running!"}


@app.post("/api/voices/train")
async def train_voice(name: str = Form(...), files: list[UploadFile] = File(...)):
    """استقبال عينات صوتية وتخزينها"""
    voice_id = str(uuid.uuid4())
    merged_path = UPLOAD_DIR / f"{voice_id}.wav"

    combined = AudioSegment.empty()

    for f in files:
        data = await f.read()
        temp_path = UPLOAD_DIR / f.filename
        with open(temp_path, "wb") as tmp:
            tmp.write(data)
        # دمج الملفات الصوتية
        audio = AudioSegment.from_file(temp_path)
        combined += audio

    combined.export(merged_path, format="wav")
    voices[voice_id] = {"name": name, "path": str(merged_path)}

    return {"voice_id": voice_id, "name": name, "status": "ready"}


@app.post("/api/tts/generate")
async def generate_tts(voice_id: str = Form(...), text: str = Form(...)):
    """تحويل النص إلى صوت باستخدام الصوت المرفوع"""
    if voice_id not in voices:
        return JSONResponse({"error": "Voice not found"}, status_code=404)

    voice_ref = voices[voice_id]["path"]
    output_path = OUTPUT_DIR / f"{uuid.uuid4()}.wav"

    # توليد الصوت
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_ref,
        language="ar",
        file_path=output_path
    )

    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")


@app.get("/api/health")
def health_check():
    return {"status": "running"}
