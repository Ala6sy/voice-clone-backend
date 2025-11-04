# FILE: main.py
# -------------------------------------------
# FastAPI Backend for Voice Cloning Lab (نسخة خفيفة بدون TTS)
# -------------------------------------------
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
from pydub import AudioSegment

# إنشاء التطبيق
app = FastAPI(title="Voice Cloning Lab API (Light Version)")

# إعداد CORS للسماح بالوصول من الواجهة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # يمكنك تضييقها لاحقًا على نطاق موقعك فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مجلدات التخزين
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ذاكرة مؤقتة للأصوات المسجّلة (في RAM فقط)
voices: dict[str, dict] = {}


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Voice Cloning Backend running! (light, no TTS yet)",
    }


@app.get("/api/health")
def health_check():
    return {"status": "running"}


@app.post("/api/voices/train")
async def train_voice(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    استقبال عينات صوتية من المستخدم ودمجها في ملف واحد
    وتخزين المسار في الذاكرة المؤقتة.
    """
    voice_id = str(uuid.uuid4())
    merged_path = UPLOAD_DIR / f"{voice_id}.wav"

    combined = AudioSegment.empty()

    for f in files:
        # قراءة بيانات الملف
        data = await f.read()
        temp_path = UPLOAD_DIR / f.filename

        # حفظ الملف المؤقت
        with open(temp_path, "wb") as tmp:
            tmp.write(data)

        # تحميله بـ pydub ودمجه مع الباقي
        audio = AudioSegment.from_file(temp_path)
        combined += audio

    # حفظ الملف المدمج النهائي
    combined.export(merged_path, format="wav")

    # تخزين بيانات الصوت في الذاكرة
    voices[voice_id] = {"name": name, "path": str(merged_path)}

    return {
        "voice_id": voice_id,
        "name": name,
        "status": "ready",
    }


@app.post("/api/tts/generate")
async def generate_tts(
    voice_id: str = Form(...),
    text: str = Form(...),
):
    """
    نسخة تجريبية خفيفة:
    الآن لا يوجد توليد صوت حقيقي، فقط نتحقق من الصوت
    ونرجع استجابة JSON. الهدف هو فقط تشغيل السيرفر على Render.
    """
    if voice_id not in voices:
        return JSONResponse({"error": "Voice not found"}, status_code=404)

    voice_ref = voices[voice_id]["path"]

    # هنا مستقبلاً سنستدعي خدمة TTS حقيقية (على سيرفر أقوى أو API خارجي)
    # حالياً نرجع معلومات بسيطة للتجربة
    return {
        "status": "ok",
        "message": "TTS is not implemented on this free Render server yet.",
        "voice_id": voice_id,
        "voice_path": voice_ref,
        "text": text,
    }
