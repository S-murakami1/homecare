from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from make_report import make_report
from make_text import transcribe_audio
from loguru import logger
import traceback

app = Flask(__name__)
CORS(app)

# アップロード設定
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'m4a', 'mp3', 'wav', 'mp4'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB制限

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_audio_file(file_path):
    """main.pyと同じ処理を実行"""
    try:
        logger.info(f"ファイル処理開始: {file_path}")
        logger.info(f"ファイルサイズ: {os.path.getsize(file_path)} bytes")
        logger.info(f"ファイル拡張子: {os.path.splitext(file_path)[1]}")
        
        transcript = transcribe_audio(file_path)
        logger.info(f"transcript: {transcript}")
        report = make_report(transcript)
        logger.info(f"report: {report}")
        return transcript, report
    except Exception as e:
        logger.error(f"process_audio_file エラー: {str(e)}")
        logger.error(f"エラーの詳細: {traceback.format_exc()}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error("ファイルがリクエストに含まれていません")
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("ファイル名が空です")
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        logger.info(f"アップロードされたファイル: {file.filename}")
        logger.info(f"ファイルのMIMEタイプ: {file.content_type}")
        logger.info(f"ファイルサイズ: {len(file.read())} bytes")
        file.seek(0)  # ファイルポインタをリセット
        
        if file and allowed_file(file.filename):
            # 元のファイル名から拡張子を取得
            original_ext = os.path.splitext(file.filename)[1].lower()
            # 安全なファイル名を生成（拡張子を保持）
            safe_name = secure_filename(file.filename)
            # 拡張子が失われた場合は追加
            if not safe_name.endswith(original_ext):
                safe_name = f"audio{original_ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            file.save(filepath)
            
            logger.info(f"ファイル保存完了: {filepath}")
            
            # main.pyと同じ処理を実行
            logger.info(f"音声ファイルを処理中: {safe_name}")
            transcript, report = process_audio_file(filepath)
            
            # 一時ファイルを削除
            os.remove(filepath)
            logger.info(f"一時ファイル削除完了: {filepath}")
            
            return jsonify({
                'success': True,
                'transcript': transcript,
                'report': report
            })
        else:
            logger.error(f"許可されていないファイル形式: {file.filename}")
            return jsonify({'error': '許可されていないファイル形式です'}), 400
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"エラーが発生しました: {str(e)}")
        logger.error(f"エラーの詳細: {error_details}")
        return jsonify({
            'error': f'処理中にエラーが発生しました: {str(e)}',
            'details': error_details
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 