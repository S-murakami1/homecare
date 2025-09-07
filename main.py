from make_report import make_report
from make_text import transcribe_audio
from loguru import logger


file_path = "./皮下点滴_田中一郎.m4a"

if __name__ == "__main__":
    transcript = transcribe_audio(file_path)
    logger.info(f"transcript: {transcript}")
    report = make_report(transcript)
    logger.info(f"report: {report}")