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
## S
- 患者は酸素飽和度が98%であることに驚いている様子。
- 飲水時にむせ込みがあることを認識しており、座位での飲水の重要性を理解している。
- 点滴時に多少の痛みがあることを報告。

## O
- 酸素飽和度は98%。
- とろみをつけた水分でむせ込みが少なくなっている。
- 点滴は5～6時間継続予定。

## A
- 酸素飽和度が98%と高く、肺炎の改善が見られる。
- とろみをつけた水分による飲水でむせ込みが軽減。
- 点滴中に多少の痛みがあるが、患者は耐えられる範囲内。

## P
- 患者に座位での飲水を指導し、むせ込みの軽減を図る。
- 点滴の痛みについては細心の注意を払い、患者の快適さを優先する。
- 次回の訪問時までに点滴が終了するようスケジュールを調整。

------------------------
## Summary
患者は現在の酸素飽和度が98%であり、肺炎の状態が改善していることが確認できました。とろみを加えた水分を摂取することで、むせ込みが軽減されており、今後は座位での飲水を推奨しています。また、点滴中に多少の痛みがあることがわかりましたが、患者は耐えられる範囲内としています。点滴は指定時 間内に終える予定です。
"""
    sentence2 = """
## S
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

------------------------
## Summary
SpO2は酸素送気なしで90％台後半を示しており、発熱などの感染兆候もない。肺炎は改善傾向だが、「咳が止まらない」と、浅いギャッジアップのまま飲水をしてしまうことで咳嗽反射が見られている。飲水時の姿勢に注意するように利用者本人へ説明する。活気はあるが、引き続き誤嚥性肺炎の再発に注意する。
腹部皮下に留置針挿入し皮下点滴実施する。15時終了予定で、開始前・開始時には皮下点滴穿刺部位にトラブルはない。「点滴すると安心するもんな」と話されており、不安や苦痛なく点滴を継続できるように今後も皮膚トラブル等に注意して観察していく。

"""

    similarity = calc_similarity(sentence1, sentence2)
    logger.success(f"文章1と文章2の類似度: {similarity}")

    ref = normalize_for_bleu(sentence2)
    hyp = normalize_for_bleu(sentence1)

    bleu = list_bleu([ref], [hyp])  # 行数1 vs 1 なのでOK
    logger.success(f"BLEU: {bleu}")
