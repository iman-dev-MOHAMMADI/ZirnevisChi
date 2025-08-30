import torch
from pyannote.audio import Pipeline
import os
import logging

# --- Logging Setup ---
logger = logging.getLogger(__name__)

def format_time(seconds):
    """
    Convert seconds to HH:MM:SS.mmm string format.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time string.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

def diarize_audio(audio_path, auth_token):
    """
    Perform speaker diarization on an audio file using Pyannote.

    Args:
        audio_path (str): Path to the audio file (16kHz mono WAV recommended).
        auth_token (str): Hugging Face authentication token.

    Returns:
        list[dict]: List of diarization segments with 'start', 'end', and 'speaker' keys.
    """
    if not auth_token:
        logger.error("Hugging Face authentication token is missing.")
        raise ValueError("Hugging Face token is required for diarization.")

    logger.info("Instantiating speaker diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token
    )
    logger.info("Pipeline instantiated successfully.")

    if torch.cuda.is_available():
        logger.info("Moving pipeline to GPU.")
        pipeline.to(torch.device("cuda"))
    else:
        logger.info("GPU not available, using CPU for processing.")

    logger.info(f"Running speaker diarization on {audio_path}...")
    diarization = pipeline(audio_path)
    logger.info("Diarization processing complete.")

    diarization_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_segments.append({
            'start': float(turn.start),
            'end': float(turn.end),
            'speaker': str(speaker)
        })
    logger.info("Diarization output formatted successfully.")
    return diarization_segments

if __name__ == "__main__":
    # This section is only for direct testing of this module.
    # For full application, run the bot from main.py.
    from dotenv import load_dotenv
    load_dotenv()

    # --- Test variables ---
    # Place your test audio file (wav, 16kHz) in the processed_files folder
    input_audio_file = "processed_files/your_test_audio_file.wav"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    if not os.path.exists(input_audio_file):
        print(f"Error: Input file not found at '{input_audio_file}'")
    elif not huggingface_token:
        print("Error: Hugging Face token not found. Set HUGGINGFACE_TOKEN in your .env file.")
    else:
        try:
            print("Starting diarization test...")
            diarization_result = diarize_audio(input_audio_file, huggingface_token)
            
            output_file = "diarization_output_test.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(str(diarization_result))
            
            print("\nDiarization test completed successfully.")
            print(f"Result saved to {output_file}")

        except Exception as e:
            print(f"An error occurred during diarization test: {e}")
