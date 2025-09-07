import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import faiss

# 1. SQLiteデータベースの準備
def create_faq_database():
    conn = sqlite3.connect("faq.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS faq (id INTEGER PRIMARY KEY, question TEXT)")
    cursor.execute("DELETE FROM faq")

    # サンプルデータを追加
    sample_faqs = [
        "How do I reset my password?",
        "What payment methods do you accept?",
        "Can I return an item I bought?",
        "How can I track my order?"
    ]

    cursor.executemany("INSERT INTO faq (question) VALUES (?)", [(q,) for q in sample_faqs])
    conn.commit()
    return conn

# 2. 文章の埋め込みの作成
def compute_faq_embeddings(faq_questions, model):
    return np.array(model.encode(faq_questions))

# 3. Faissインデックスの作成
def create_faiss_index(model, embeddings, doc_ids):
    dimension = model.get_sentence_embedding_dimension()
    index_flat_l2 = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index_flat_l2)
    index.add_with_ids(embeddings, np.array(doc_ids)) # Convert doc_ids to numpy array
    return index, index_flat_l2

# 4. クエリ処理
def search_faq(query, model, index, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    logger.info(f"Distances: {distances}")
    logger.info(f"Indices: {indices}")
    return indices[0]

# 5. 結果の取得
def get_faq_results(faq_indices, conn):
    cursor = conn.cursor()
    # Retrieve the actual faq_ids from the index results
    faq_ids = [int(idx) for idx in faq_indices]
    cursor.execute("SELECT * FROM faq WHERE id IN ({})".format(",".join("?" * len(faq_ids))), faq_ids)
    return cursor.fetchall()

# データベース作成
conn = create_faq_database()

# sentence-transformersモデルを読み込む
# model = SentenceTransformer("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
model = SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1")

# FAQデータを読み込み、埋め込みベクトルを計算
cursor = conn.cursor()
cursor.execute("SELECT id, question FROM faq")
faq_data = cursor.fetchall()
logger.info(f"FAQ data: {faq_data}")
faq_ids, faq_questions = zip(*faq_data)
faq_embeddings = compute_faq_embeddings(faq_questions, model)

# Faissインデックスを作成
index, index_flat_l2 = create_faiss_index(model, faq_embeddings, faq_ids)

# クエリを入力して検索
query = "How can I change my password?"
faq_indices = search_faq(query, model, index) # Use 'index' instead of 'index_flat_l2'

# 結果を取得して表示
results = get_faq_results(faq_indices, conn)
logger.info("Results for query:", query)
for r in results:
    logger.info(f"ID: {r[0]}, Question: {r[1]}")

# データベース接続を閉じる
# conn.close()