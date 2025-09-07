from sentence_transformers import SentenceTransformer, util
from loguru import logger
from bleu import list_bleu

# モデルのロード
model = SentenceTransformer('stsb-xlm-r-multilingual')
model.max_seq_length = 512

# トークナイザ取得と切り捨て判定ヘルパ
first = model._first_module()
tok = first.tokenizer

def is_truncated(s: str):
    ids_full = tok.encode(s, add_special_tokens=True, truncation=False)
    ids_trunc = tok.encode(s, add_special_tokens=True, truncation=True, max_length=model.get_max_seq_length())
    return len(ids_full) > len(ids_trunc), len(ids_full), len(ids_trunc)

def log_truncation_status(name: str, s: str):
    truncated, tokens_full, tokens_used = is_truncated(s)
    return truncated

def calc_similarity(sentence1, sentence2):
    # 文章をベクトルに変換（事前に切り捨てチェックしてログ）
    tr1 = log_truncation_status("sentence1", sentence1)
    tr2 = log_truncation_status("sentence2", sentence2)
    if tr1 or tr2:
        logger.warning("トークン切り捨てが発生しました。")

    # コサイン類似度の計算
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)

    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]

    # SentenceTransformer内部のトークナイザを取得
    first = model._first_module()
    tok = first.tokenizer
    return cosine_score

def normalize_for_bleu(t: str) -> str:
    # 改行や余分な空白を潰して1行化
    return " ".join(ln.strip() for ln in t.splitlines() if ln.strip())

if __name__ == "__main__":
    # 比較する文章
    sentence1 = """
## summary
患者は肺炎が改善しており、酸素飽和度は98%を示している。咳嗽が続いているが、飲水時のむせ込みは改善している。点滴中の 軽度の痛みを訴えたが、今後は口からの水分摂取を促す計画。

## S
患者は「咳が嗽が少し続いているが、改善してきた」と述べ、飲水時のむせ込みが「水分にとろみをつけた」で減少したと報告した。また、点滴中に「少し痛みがあった」と訴えたが、全体的に安定している様子が見受けられた。

## O
- 酸素飽和度: 98%
- 肺炎の改善が見られ、咳嗽は持続中。
- 水分摂取時のむせ込みは改善傾向。
- 点滴中に軽度の痛み（患者の主観により）。
- 口からの水分摂取が進めば点滴を終了予定。

## A
患者は肺炎の状態が改善し、酸素飽和度も正常範囲内であるが、咳嗽と飲水時のむせ込みが依然として存在する。また、点滴に伴う痛みは軽度であり、全体的には安定している。口からの水分摂取を促進し、自然な水分摂取ができるよう支援することが重要である。

## P
- 引き続き、酸素管理を行い、酸素飽和度を維持。
- 今日の点滴は5時間から6時間実施予定。次回訪問時に終了予定。
- 飲水時の体勢指導を行い、座位での水分摂取を促進。
- 点滴による痛みが強くなる場合は、看護師によるフォローアップを行う。
"""
    sentence2 = """
##S
「咳が止まらない」「点滴打つと安心するもんな」
## O
- SpO2はルームエアーで98％、活気がある
- 発熱などの肺炎悪化の徴候なし
- ベッド上で浅いギャッジアップのまま飲水し咳嗽反射が見られる
- 腹部に皮下点滴実施、刺入部トラブルなし。
## A
- 誤嚥性肺炎は改善傾向、活気があり、咳嗽反射も正常に見られている。
## P
- このまま肺炎の悪化がないか経過をみていく
- 誤嚥を再発しないように飲水時の姿勢について指導した
- 皮下点滴は15時投与終了予定。今後も苦痛少なく皮下点滴を継続できるように皮膚トラブルや点滴トラブルに注意して実施する。
"""

    similarity = calc_similarity(sentence1, sentence2)
    logger.success(f"文章1と文章2の類似度: {similarity}")

    ref = normalize_for_bleu(sentence2)
    hyp = normalize_for_bleu(sentence1)

    bleu = list_bleu([ref], [hyp])  # 行数1 vs 1 なのでOK
    logger.success(f"BLEU: {bleu}")
