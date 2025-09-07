import os
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class HomecareSummary(BaseModel):
    summary: str
    subjects: Optional[List[str]] = None
    objects: List[str]
    assessments: Optional[List[str]] = None
    plans: List[str]


def get_response(transcript: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""
    ## 指示
    あなたは訪問看護の記録作成支援AIです。以下の"## 会話文字起こし"から、日本語で看護記録の要約を作成してください。

    ## 制約条件
    - 適宜専門的な用語を使用してください。

    ## 出力形式
    summary: 要約（3～5行程度、重要点を簡潔に）
    subjects: 主観情報
    objects: 客観情報（観察所見・処置内容を箇条書き）
    assessments: 評価（看護上の解釈・問題点。明確な訴えが無ければ「なし」）
    plans: 計画（今後の対応や指導、観察継続点）

    ## 会話文字起こし
    {transcript}
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript}
        ],
        response_format=HomecareSummary
    )
    return response.choices[0].message.content

def make_report(transcript: str) -> str:
    # response = get_response(transcript)
    # report = HomecareSummary.model_validate_json(response)
    # past_record_path = "./records.txt"
    new_record_path = "./new_record.txt"
    
    #try:
    #    with open(past_record_path, "r") as f:
    #         past_record = f.read()
    # except FileNotFoundError:
    #     past_record = ""
        
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""
    ## 指示
    以下の"## 会話文字起こし"から、新しい看護記録をSOAP形式で作成してください。

    ## 制約条件
    - 適宜専門的な用語を使用してください。
    - マークダウン形式で出力してください

    ## 出力形式
    ## summary
    XXX
    ## S
    XXX
    ## O
    XXX
    ## A
    XXX
    ## P
    XXX

    ## 会話文字起こし
    {transcript}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは訪問看護の記録作成支援AIです。"},
            {"role": "user", "content": prompt}
        ],
    )
    new_record = response.choices[0].message.content
    logger.info(f"new_record: {type(new_record)}")
    return str(new_record)

def make_report_local(transcript: str) -> str:
    model = AutoModelForCausalLM.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0", device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    prompt = f"""
    ## 指示
    以下の"## 会話文字起こし"から、新しい看護記録をSOAP形式で作成してください。

    ## 制約条件
    - 適宜専門的な用語を使用してください。
    - マークダウン形式で出力してください

    ## 出力形式
    ## summary
    XXX
    ## S
    XXX
    ## O
    XXX
    ## A
    XXX
    ## P
    XXX

    ## 会話文字起こし
    {transcript}
    """
    messages = [
        {"role": "system", "content": "あなたは訪問看護の記録作成支援AIです。"},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=1024,
        temperature=0.5,
        streamer=streamer
    )
    logger.info(f"output_ids: {type(output_ids)}")
    return output_ids

if __name__ == "__main__":
    with open("script.txt", "r") as f:
        transcript = f.read()
    report = make_report(transcript)
    logger.info(f"report: {report}")