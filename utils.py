import os
import requests
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from tqdm import tqdm
import time
import uuid
import threading
import logging

class VerboseLogger:
    """Custom logger with verbosity levels"""
    
    LEVELS = {
        0: logging.ERROR,      # Only errors and critical messages
        1: logging.WARNING,    # Warnings and above
        2: logging.INFO,       # General info and above
        3: logging.DEBUG       # Detailed debug info
    }
    
    def __init__(self, verbosity=1):
        self.verbosity = verbosity
        self.logger = logging.getLogger('STT_Processor')
        
        # Configure logger if it has no handlers
        if not self.logger.handlers:
            level = self.LEVELS.get(verbosity, logging.INFO)
            self.logger.setLevel(level)
            
            # Create console handler with formatting
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Add file handler for debug level
            file_handler = logging.FileHandler('stt_processor.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)


class STTProcessor:
    """Enhanced Speech-to-Text processor based on the GoogleSTT class"""
    supported_formats = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
    temp_dir = "temp_audio_segments"

    @staticmethod
    def transcribe_audio(file_path, language='en', logger=None, timeout=300, max_retry=3):
        """Generate transcription using Google Speech Recognition"""
        if logger is None:
            logger = VerboseLogger(0).logger
            
        try:
            # Load audio file and normalize it
            logger.debug(f"Loading audio file: {file_path}")
            audio_segment = AudioSegment.from_file(str(file_path))
            audio_segment = audio_segment.set_channels(1)  # Convert to mono
            audio_segment = audio_segment.set_frame_rate(16000)  # Set sample rate to 16kHz

            # Create temp directory if needed
            temp_dir = Path(STTProcessor.temp_dir)
            temp_dir.mkdir(exist_ok=True)

            # Get audio duration
            duration_ms = len(audio_segment)
            duration_sec = duration_ms / 1000

            logger.debug(f"Transcribing segment of {duration_sec:.2f} seconds")

            # Skip very short segments
            if duration_ms < 800:
                logger.debug("Segment too short, skipping transcription")
                return {"type": "error", "text": "Segment too short", "duration": duration_sec}

            text = STTProcessor._transcribe_segment(file_path, audio_segment, language, logger, timeout, max_retry)
            if not text:
                return {"type": "error", "text": "Could not transcribe audio", "duration": duration_sec}

            return {"type": "text", "text": text, "duration": duration_sec}

        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
            return {"type": "error", "text": str(e), "duration": 0}

    @staticmethod
    def _transcribe_segment(file_path, audio_segment, language, logger=None, timeout=120, max_retry=3):
        """Transcribe a single audio segment with timeout and retries"""
        if logger is None:
            logger = VerboseLogger(0).logger
            
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8

        if timeout:
            try:
                r.operation_timeout = timeout
            except AttributeError:
                logger.warning("Installed version of speech_recognition library does not support 'operation_timeout'.")
        
        thread_id = threading.get_ident()
        uuid_str = uuid.uuid4().hex
        unique_suffix = f"{thread_id}_{uuid_str}"

        temp_path = Path(STTProcessor.temp_dir) / f"temp_segment_{unique_suffix}.wav"
        os.makedirs(STTProcessor.temp_dir, exist_ok=True)

        try:
            logger.debug(f"Exporting temp segment to {temp_path}")
            audio_segment.export(str(temp_path), format="wav")

            with sr.AudioFile(str(temp_path)) as source:
                audio_data = r.record(source)
                for attempt in range(max_retry):
                    try:
                        return r.recognize_google(audio_data, language=language)
                    except sr.UnknownValueError:
                        logger.debug("Google Speech Recognition could not understand audio")
                        return ""
                    except sr.WaitTimeoutError:
                        logger.warning(f"Attempt {attempt + 1}/{max_retry}: Transcription timed out.")
                        if attempt < max_retry - 1:
                            time.sleep(2 ** attempt)
                        else:
                            logger.error(f"Failed to transcribe after {max_retry} attempts due to timeout.")
                            return ""
                    except sr.RequestError as e:
                        logger.warning(f"Attempt {attempt + 1}/{max_retry}: Could not request results; {e}")
                        if attempt < max_retry - 1:
                            time.sleep(2 ** attempt)
                        else:
                            logger.error(f"Failed to transcribe after {max_retry} attempts.")
                            return ""
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting temp file {temp_path}: {e}")


class AudioDownloader:
    """Downloads audio files from URLs"""

    def __init__(self, download_dir="downloaded_audio", logger=None):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.logger = logger or VerboseLogger(0).logger

    def download_file(self, url, source_name, filename=None, timeout=300):
        """Download a file from a URL with timeout"""
        if filename is None:
            filename = f"{source_name}.mp3"

        file_path = self.download_dir / filename

        if file_path.exists():
            self.logger.info(f"File {filename} already exists. Skipping download.")
            return {"path": str(file_path), "success": True, "filename": filename}

        try:
            self.logger.info(f"Downloading {url} to {file_path}")
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, disable=self.logger.level > logging.INFO) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            return {"path": str(file_path), "success": True, "filename": filename, "source": source_name}
        except requests.Timeout:
            self.logger.error(f"Timeout downloading {url}")
            return {"path": None, "success": False, "error": f"Download timeout after {timeout} seconds", "source": source_name}
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            return {"path": None, "success": False, "error": str(e), "source": source_name}


class AudioSegmenter:
    """Segments audio files based on silence detection"""

    def __init__(self, output_dir="segmented_audio", logger=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger or VerboseLogger(0).logger

    def segment_audio(self, file_path, source_id, min_silence_len=300, silence_thresh=-35, max_segment_len=10000, target_len=5000):
        """Segment audio file based on silence"""
        file_path = Path(file_path)
        self.logger.info(f"Loading audio file for segmentation: {file_path.name}")
        audio = AudioSegment.from_file(str(file_path))

        self.logger.info(f"Splitting {file_path.name} on silence...")
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=200
        )

        if not chunks:
            self.logger.info("No natural breaks found. Using time-based chunking.")
            chunks = [audio[i:i + target_len] for i in range(0, len(audio), target_len)]

        processed_chunks = []
        for chunk in chunks:
            if len(chunk) < 500: continue # Skip very short chunks
            if len(chunk) > max_segment_len:
                sub_chunks = [chunk[i:i + target_len] for i in range(0, len(chunk), target_len)]
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(chunk)

        segment_paths = []
        self.logger.info(f"Found {len(processed_chunks)} segments to export")

        for i, chunk in enumerate(processed_chunks):
            segment_path = self.output_dir / f"{source_id}_segment_{i + 1}.mp3"
            chunk.export(str(segment_path), format="mp3", parameters=["-ac", "1", "-ar", "16000"])
            segment_paths.append(str(segment_path))

        return segment_paths