import os
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
from loguru import logger


img_path = "./20130215_358292.jpg"

def encode_image_to_base64(img_path):
    with Image.open(img_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image(img_path):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # 画像をbase64エンコード
    base64_image = encode_image_to_base64(img_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "あなたは経験豊富な専門医です。医療の専門家として、画像を詳しく分析し、医学的な見地から説明してください。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "この画像について医学的な観点から鑑別を上げて下さい。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    import glob
    import os

    data_dir = "./data"
    img_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    result = 
    for img_path in img_paths:
        logger.info(f"img_path: {img_path}")
        result = analyze_image(img_path=img_path)
        logger.info(f"result: {result}")
