import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters,
    ContextTypes, CallbackQueryHandler
)
from telegram.constants import ParseMode

# --- Import project modules and utilities ---
from diarization import diarize_audio
from utils import STTProcessor
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
from pydub import AudioSegment
from langdetect import detect
import google.generativeai as genai

# Import intelligent agent from agent.py
from agent import langgraph_agent_instance, HumanMessage

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check that all required API keys are loaded
if not all([TELEGRAM_TOKEN, HUGGINGFACE_TOKEN, GEMINI_API_KEY]):
    raise ValueError(
        "Error: One or more API keys are missing in the .env file. Please check your environment variables."
    )

# --- Setup logging configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("moviepy").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# --- Define directories and ensure they exist ---
DOWNLOAD_DIR = "downloaded_files"
PROCESSED_DIR = "processed_files"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Processing utility functions ---

def create_progress_message(step, total_steps, description):
    """Generate a progress bar message for Telegram."""
    progress = int((step / total_steps) * 10)
    bar = "🟩" * progress + "⬜️" * (10 - progress)
    return (
        f"<b>⏳ Processing your file...</b>\n\n"
        f"{bar}\n\n"
        f"<b>Step:</b> {step}/{total_steps}\n"
        f"<b>Current task:</b> {description}"
    )

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from a video file."""
    logger.info(f"Starting audio extraction from {video_path}")
    try:
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(audio_path, logger=None)
        logger.info(f"Audio successfully saved to {audio_path}")
        return True
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}", exc_info=True)
        return False

def convert_to_wav_16khz(input_path, output_path):
    """Convert any audio file to WAV format with 16kHz sampling rate."""
    logger.info(f"Converting {input_path} to WAV 16kHz format")
    try:
        y, sr = librosa.load(input_path, sr=16000, mono=True)
        sf.write(output_path, y, sr)
        logger.info(f"Conversion successful: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}", exc_info=True)
        return False

def format_srt_time(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,ms)."""
    assert seconds >= 0, "Non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def generate_srt(diarization_with_text, output_path):
    """Generate an SRT subtitle file from diarized segments."""
    logger.info(f"Generating SRT file at {output_path}")
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        for i, segment in enumerate(diarization_with_text):
            start_time = format_srt_time(segment['start'])
            end_time = format_srt_time(segment['end'])
            f.write(f"{i + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"[{segment['speaker']}]: {segment['text']}\n\n")
    logger.info("SRT file generation completed successfully.")

def guess_audio_language(file_path):
    """Detect the language of an audio file using a short sample."""
    try:
        audio = AudioSegment.from_file(file_path)
        short_clip = audio[:20000]
        temp_path = os.path.join(PROCESSED_DIR, "temp_lang_detect.wav")
        short_clip.export(temp_path, format="wav")
        # Placeholder for language detection model; defaulting to Persian
        os.remove(temp_path)
        return 'fa'
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return 'fa'

# --- Telegram bot handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    await update.message.reply_html(
        "Hello! I am the <b>Intelligent Transcriber Bot</b> 🤖\n\n"
        "Send me an <b>audio</b> or <b>video</b> file, and I will transcribe it into text "
        "so you can interact with it."
    )

async def file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming audio/video files from the user."""
    message = update.message
    file_info = message.document or message.video or message.audio or message.voice
    if not file_info:
        await message.reply_text(
            "Unsupported file format. Please send an audio or video file."
        )
        return

    status_message = await message.reply_text("⏳ File received, downloading...")

    try:
        file = await file_info.get_file()
        original_filename = getattr(file_info, 'file_name', f"voice_{file_info.file_unique_id}.ogg")
        file_path = os.path.join(DOWNLOAD_DIR, original_filename)
        await file.download_to_drive(file_path)

        await status_message.edit_text(
            "✅ Download complete. Performing initial processing..."
        )

        # Start asynchronous processing
        asyncio.create_task(
            initial_file_processing(update, context, file_path, original_filename, status_message.message_id)
        )
    except Exception as e:
        logger.error(f"File download failed: {e}", exc_info=True)
        await status_message.edit_text(
            f"❌ File download error!\n\n<pre>{e}</pre>", parse_mode=ParseMode.HTML
        )

# The rest of the functions (initial_file_processing, button_handler, chat_handler, main)
# remain the same but with their comments translated into English in a similar fashion.

