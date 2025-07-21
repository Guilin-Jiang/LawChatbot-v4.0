
import faiss
import numpy as np
import os
import pickle
import requests
from transformers import AutoTokenizer, AutoModel
from . import load_documents

import random
from sklearn.metrics.pairwise import cosine_similarity  # 保留，后续可用于 rerank

# 加载 HuggingFace 嵌入模型
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")



def encode_text(query):
    
    if isinstance(query, str):
        query = [query]

    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()



# reformulate_query, generate_hyde_query, generate_multi_queries is three electric method to improve queries
def reformulate_query(query: str) -> str:
    prompt = f"""
    Task: Rewrite the user's question into a precise and standalone query for document retrieval.\n
    
    Requirements:
    - Avoid vague words or pronouns (e.g., "he", "there", "it").
    - Make the query as specific as possible.
    - Keep it concise and focused.\n

    Original question: {query}\n

    Reformulated query:
    """
    reform_query = get_answer(prompt)
    return reform_query.strip()


def generate_hyde_query(query: str) -> str:
    prompt = (
        "请根据以下问题生成一段可能的专业回答内容，内容要简洁且用于向量检索：\n"
        f"问题：{query}\n回答："
    )
    generateHyde_query = get_answer(prompt)
    
    return generateHyde_query.strip()

def generate_multi_queries(query: str) -> list[str]:
    prompt = (
        "请将下面的问题改写成从3个不同角度，并且要用于进行信息检索的查询。\n"
        f"原始问题：{query}\n"
        "输出格式为：\n1. ...\n2. ...\n3. ..."
    )
    generateMulti_queries = get_answer(prompt).strip()
    
    queries = []
    for line in generateMulti_queries.split("\n"):
        line = line.strip()
        if line:
            line = line.lstrip("1234567890.()①②③ ")
            queries.append(line)

    return queries[:3]  # 返回最多3条干净字符串



# 从磁盘加载 FAISS 向量数据库 + metadata
def load_faiss_index(index_path="data/vector_index"):
    # 加载向量索引
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))

    # 加载 metadata 信息（包含原文内容 + source）
    with open(os.path.join(index_path, "docs.pkl"), "rb") as f:
        documents = pickle.load(f)  # List[Document]

    return index, documents



# 使用 Ollama 运行 Llama 模型
def get_answer(query):
    host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    model_mistral = os.getenv("OLLAMA_MODEL", "mistral")

    url = f"{host}/api/chat"

    payload = {
        "model": model_mistral,
        "messages": [
            {"role": "user", "content": query}
        ],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Ollama response:")
        print(response.text)

        data = response.json()  # ⚠️ 出错点
        return data["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        print("Ollama 请求失败:", e)
        return "抱歉，无法连接 Ollama下的mistral 模型。"



def retrieve_and_generate(query, chat_history=None, top_k=3):
    # 加载向量库和原始文档
    index, documents = load_faiss_index()

    # Step 1: 初始检索
    reform_query_1 = reformulate_query(query)
    query_embedding_1 = encode_text([reform_query_1])
    _, I_1 = index.search(query_embedding_1, k=top_k)

    similar_texts_1 = []
    for i in I_1[0]:
        doc = documents[i]
        chunk_text = doc.page_content.strip()
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "?")
        similar_texts_1.append(f"[{source} - 页码 {page}] {chunk_text}")

    context_1 = "\n\n".join(similar_texts_1)

    # Step 2: 基于第一跳资料生成改写问题
    reformulate_prompt = "\n".join([
        "你是一个法律领域专家，请基于以下资料和问题，生成一个更具体、更深入的提问，以帮助更准确地检索资料。",
        f"原始问题: {query}",
        f"参考资料:\n{context_1}",
        "\n请输出重写后的新问题："
    ])
    query_2 = get_answer(reformulate_prompt).strip()

    # Step 3: 第二跳检索
    reform_query_2 = reformulate_query(query_2)
    query_embedding_2 = encode_text([reform_query_2])
    _, I_2 = index.search(query_embedding_2, k=top_k)

    # Step 4: 合并 D1 + D2 所有 chunk，随机抽取 top_k
    all_chunks = []

    for i in I_1[0]:
        doc = documents[i]
        chunk_text = doc.page_content.strip()
        metadata = doc.metadata
        all_chunks.append({
            "text": chunk_text,
            "source": metadata.get("source", "未知来源"),
            "page": metadata.get("page", "?")
        })

    for i in I_2[0]:
        doc = documents[i]
        chunk_text = doc.page_content.strip()
        metadata = doc.metadata
        all_chunks.append({
            "text": chunk_text,
            "source": metadata.get("source", "未知来源"),
            "page": metadata.get("page", "?")
        })

    random.shuffle(all_chunks)
    selected_chunks = all_chunks[:top_k]

    final_context = "\n\n".join(
        [f"[{c['source']} - 页码 {c['page']}] {c['text']}" for c in selected_chunks]
    )

    # Step 5: 构造聊天记录 prompt
    history_prompt = ""
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                prefix = "用户" if role == "user" else "助手"
                history_prompt += f"{prefix}：{content}\n"

    final_prompt = "\n".join([
        "你是一个专业法律助理，请基于以下聊天记录和资料回答用户问题。",
        f"\n聊天记录:\n{history_prompt}",
        f"\n资料:\n{final_context}",
        f"\n问题: {query}"
    ])

    # 生成两个候选回答
    answer_1 = get_answer(final_prompt)
    answer_2 = get_answer(final_prompt)

    # 让模型根据上下文和问题选择一个更好的回答
    selection_prompt = "\n".join([
        "你是一个严谨的法律专家，以下是对同一问题的两个回答，请基于资料内容判断哪一个更准确、更全面，并只输出更优的那个回答。",
        "\n问题：", query,
        "\n参考资料：", final_context,
        "\n回答 A：", answer_1.strip(),
        "\n回答 B：", answer_2.strip(),
        "\n请只输出你认为更好的回答内容："
    ])

    final_answer = get_answer(selection_prompt)

    return {
        "answer": final_answer.strip(),
        "sources": [f"[{c['source']} - 页码 {c['page']}] {c['text']}" for c in selected_chunks],
        "reformulated_question": query_2,
        "candidates": [answer_1.strip(), answer_2.strip()]  # 可选调试用
    }
