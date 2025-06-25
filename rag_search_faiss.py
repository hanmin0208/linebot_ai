import os
import openai
import numpy as np
import faiss
import glob
import hashlib

# 設定 OpenAI 金鑰（建議用環境變數）
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# 資料與儲存設定
KNOWLEDGE_DIR = "rag_docs"
INDEX_FILE = "faiss_index.bin"
DOC_MAPPING_FILE = "doc_mapping.txt"


def get_embedding(text):
    """取得 OpenAI 向量嵌入"""
    response = openai.Embedding.create(
        model=EMBED_MODEL,
        input=text.strip().replace("\n", " ")
    )
    return np.array(response["data"][0]["embedding"], dtype=np.float32)


def hash_text(text):
    """避免重複段落（透過 md5）"""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def load_all_chunks(knowledge_dir=KNOWLEDGE_DIR):
    """從資料夾中讀取所有 txt 檔並切段落"""
    chunks, sources, hashes = [], [], set()
    files = glob.glob(os.path.join(knowledge_dir, "*.txt"))
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        parts = [p.strip() for p in content.split("\n\n") if p.strip()]
        for p in parts:
            h = hash_text(p)
            if h not in hashes:
                chunks.append(p)
                sources.append(os.path.basename(fpath))
                hashes.add(h)
    return chunks, sources


def build_or_update_faiss_index():
    """建立或更新 FAISS 向量索引"""
    chunks, sources = load_all_chunks()
    if not chunks:
        print("❌ 未找到任何知識段落")
        return

    # 建立向量矩陣
    embeddings = [get_embedding(c) for c in chunks]
    matrix = np.vstack(embeddings)

    # 建立與儲存索引
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(matrix)
    faiss.write_index(index, INDEX_FILE)

    # 儲存段落來源
    with open(DOC_MAPPING_FILE, "w", encoding="utf-8") as f:
        for chunk, src in zip(chunks, sources):
            clean_chunk = chunk.replace("\n", " ")
            f.write(f"{src}|||{clean_chunk}\n")

    print(f"✅ FAISS 索引建立完成，共 {len(chunks)} 筆")


def semantic_search(query, top_k=3):
    """查詢與輸入問題語意最接近的段落"""
    if not (os.path.exists(INDEX_FILE) and os.path.exists(DOC_MAPPING_FILE)):
        return "（查無索引資料，請先執行建立）"

    query_vec = get_embedding(query).reshape(1, -1)
    index = faiss.read_index(INDEX_FILE)
    D, I = index.search(query_vec, top_k)

    # 讀取對應段落內容
    with open(DOC_MAPPING_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    matched = []
    for i in I[0]:
        if i < len(lines):
            src, chunk = lines[i].split("|||", 1)
            matched.append(f"[{src.strip()}]\n{chunk.strip()}")

    return "\n\n".join(matched) if matched else "（查無相關內容）"


