# FILE: main.py
# -------------------------------------------
# FastAPI Backend for Voice Cloning Lab
# نسختان في ملف واحد:
# - وضع خفيف (مناسب لـ Render): يرجع الملف المدموج كصوت ناتج.
# - وضع متقدّم (محلي): يستخدم Coqui XTTS إذا كانت مكتبة TTS متاحة و ENABLE_TTS=1.
# -------------------------------------------

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict
import os
import uuid
from pydub import AudioSegment

# محاولة استيراد مكتبة TTS (قد لا تكون متوفرة على Render)
try:
    from TTS.api import TTS as CoquiTTS  # type: ignore
except Exception:  # المكتبة غير متوفرة أو حدث خطأ في الاستيراد
    CoquiTTS = None  # سنستخدم وضع خفيف بدون TTS

# -------------------------------------------
# إعداد التطبيق و CORS
# -------------------------------------------

app = FastAPI(title="Voice Cloning Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # يفضّل تقييدها على دومين موقعك لاحقاً
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------
# إعداد المجلدات
# -------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# تخزين بيانات الأصوات في الذاكرة (مؤقت)
voices: Dict[str, Dict[str, str]] = {}

# -------------------------------------------
# إعداد وضع TTS المتقدّم (اختياري – محلي)
# -------------------------------------------

ENABLE_TTS = os.getenv("ENABLE_TTS", "0") == "1" and CoquiTTS is not None
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_model = None  # سيتم تحميله عند أول طلب فقط في الوضع المتقدّم


def get_tts_model():
    """
    تحميل نموذج Coqui XTTS عند أول استخدام فقط.
    يُستخدم فقط إذا كان ENABLE_TTS=1 والمكتبة متوفّرة.
    """
    global tts_model
    if not ENABLE_TTS:
        return None

    if tts_model is None:
        # هذا الجزء ثقيل، لذلك لا يُنصح به على Render المجاني
        tts_model = CoquiTTS(
            model_name=TTS_MODEL_NAME,
            progress_bar=False,
            gpu=False,  # غيّرها إلى True لو عندك GPU محلياً
        )
    return tts_model


# -------------------------------------------
# المسارات
# -------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Voice Cloning Backend running!"}


@app.get("/api/health")
def health_check():
    """
    مسار بسيط لفحص أن الخادم يعمل.
    """
    return {
        "status": "running",
        "enable_tts": ENABLE_TTS and CoquiTTS is not None,
    }


@app.post("/api/voices/train")
async def train_voice(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    استقبال عينات صوتية متعدّدة، دمجها في ملف واحد،
    وتخزينه كمرجع للصوت.
    """
    voice_id = str(uuid.uuid4())
    merged_path = UPLOAD_DIR / f"{voice_id}.wav"

    combined = AudioSegment.empty()

    for f in files:
        data = await f.read()
        temp_path = UPLOAD_DIR / f.filename

        # حفظ الملف المؤقت
        with open(temp_path, "wb") as tmp:
            tmp.write(data)

        # قراءة الملف ودمجه
        audio = AudioSegment.from_file(temp_path)
        combined += audio

        # يمكن حذف الملف المؤقت لاحقاً إن أحببت
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # حفظ الملف المدموج النهائي
    combined.export(merged_path, format="wav")

    # تخزين بيانات الصوت في الذاكرة
    voices[voice_id] = {"name": name, "path": str(merged_path)}

    return {"voice_id": voice_id, "name": name, "status": "ready"}


@app.post("/api/tts/generate")
async def generate_tts(
    voice_id: str = Form(...),
    text: str = Form(...),
):
    """
    تحويل النص إلى صوت باستخدام الصوت المدرّب.
    - في الوضع الخفيف (Render): يعيد الملف المدموج نفسه.
    - في الوضع المتقدّم (محلي مع ENABLE_TTS=1): يولّد ملفاً جديداً باستخدام XTTS.
    """
    if voice_id not in voices:
        return JSONResponse({"error": "Voice not found"}, status_code=404)

    voice_ref = voices[voice_id]["path"]

    # محاولة استخدام TTS المتقدّم إذا كان مفعّلاً ومتاحاً
    tts = get_tts_model()

    if tts is None:
        # وضع خفيف: فقط نرجع الملف المدموج نفسه كصوت ناتج
        return FileResponse(
            voice_ref,
            media_type="audio/wav",
            filename=f"{voice_id}.wav",
        )

    # وضع متقدّم: توليد صوت فعلي من النص باستخدام XTTS
    output_path = OUTPUT_DIR / f"{uuid.uuid4()}.wav"

    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=voice_ref,
            language="ar",
            file_path=str(output_path),
        )
    except Exception as e:
        # لو حدث خطأ في التوليد، لا نسقط الخادم، بل نرجع الملف المدموج
        return JSONResponse(
            {
                "error": "TTS generation failed, fallback to raw voice file.",
                "details": str(e),
            },
            status_code=500,
        )

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="output.wav",
    )
