import os
import whisper
from openai import OpenAI
from loguru import logger

# OpenBLASの警告を抑制
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# GPU設定（オプション）
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 特定のGPUを使用する場合

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

file_path = "./レコーディング.m4a"

def transcribe_audio_local(file_path: str) -> str:
    # GPUが利用可能かチェック
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = whisper.load_model("turbo", device=device)  # GPU/CPUを自動選択
    result = model.transcribe(file_path, language="ja")
    return result["text"]


def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # もしくは whisper-1
            file=f
        )
    return transcript.text

if __name__ == "__main__":
    file_path = "./皮下点滴_田中一郎.m4a"
    text_local = transcribe_audio_local(file_path)
    text = transcribe_audio(file_path)
    logger.info(f"text_local: {text_local}")
    logger.info(f"text: {text}")
