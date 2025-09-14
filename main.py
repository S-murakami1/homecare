from make_report import make_report, make_report_local
from make_text import transcribe_audio, transcribe_audio_local
from loguru import logger
import time

file_path = "./皮下点滴_田中一郎.m4a"

if __name__ == "__main__":
    # 音声文字起こし（ローカル）の時間計測
    start = time.time()
    #transcript = transcribe_audio_local(file_path)
    transcript = transcribe_audio(file_path)
    elapsed = time.time() - start
    logger.info(f"transcript: {transcript}")
    logger.info(f"transcribe_audio_local 実行時間: {elapsed:.2f}秒")

    # OpenAIによるレポート生成の時間計測
    start = time.time()
    report = make_report(transcript)
    elapsed = time.time() - start
    logger.info(f"report: {report}")
    logger.info(f"make_report 実行時間: {elapsed:.2f}秒")

    # ローカルモデルによるレポート生成の時間計測
    start = time.time()
    report_local = make_report_local(transcript)
    elapsed = time.time() - start
    logger.info(f"report_local: {report_local}")
    logger.info(f"make_report_local 実行時間: {elapsed:.2f}秒")