import os
import pickle
import faiss
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from sentence_transformers import SentenceTransformer

# 路径配置
VECTOR_INDEX_DIR = "app/data/vector_index"
DOCS_PATH = os.path.join(VECTOR_INDEX_DIR, "docs.pkl")
INDEX_FAISS_PATH = os.path.join(VECTOR_INDEX_DIR, "index.faiss")
EMBEDDINGS_PATH = os.path.join(VECTOR_INDEX_DIR, "index.pkl")

# 加载数据
def extract_all_vectors(index):
    return np.vstack([index.reconstruct(i) for i in range(index.ntotal)])

def load_data():
    with open(DOCS_PATH, 'rb') as f:
        docs = pickle.load(f)
    index = faiss.read_index(INDEX_FAISS_PATH)
    vectors = extract_all_vectors(index)
    return docs, vectors, index


# t-SNE 降维
def compute_tsne(vectors):
    print(f"type(vectors) = {type(vectors)}")
    print(f"vectors.shape={vectors.shape}, vectors.shape[0]={vectors.shape[0]}")
    n_samples = vectors.shape[0]
    safe_perplexity = max(5, min(30, (n_samples - 1) // 3))
    print(f"n_samples={n_samples}, Using perplexity={safe_perplexity} for t-SNE")
    tsne = TSNE(n_components=3, random_state=42, perplexity=safe_perplexity)
    return tsne.fit_transform(vectors)

# 显示 t-SNE 图
def show_tsne_plot(tsne_3d, docs, highlight_idx=None, question_point=None, output_path="data/figure/tsne_plot.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        'x': list(tsne_3d[:, 0]),
        'y': list(tsne_3d[:, 1]),
        'z': list(tsne_3d[:, 2]),
        'text': [str(doc)[:100] for doc in docs],
        'color': ["highlighted" if i == highlight_idx else "normal" for i in range(len(docs))]
    }
    if question_point is not None:
        data['x'].append(question_point[0])
        data['y'].append(question_point[1])
        data['z'].append(question_point[2])
        data['text'].append("Query")
        data['color'].append("query")

    fig = px.scatter_3d(data, x='x', y='y', z='z', color='color', hover_name='text', opacity=0.7)
    fig.write_html(output_path)  # 保存为交互式 HTML 文件


# 主程序入口
if __name__ == '__main__':
    docs, vectors, index = load_data()

    # 处理 embeddings 数据
    if isinstance(vectors, dict):
        vectors_array = np.array(list(vectors.values()))
    else:
        vectors_array = np.asarray(vectors)

    tsne_3d = compute_tsne(vectors_array)

    # 用户输入问题
    question = "what is madison university law?"

    if question.strip():
        model = SentenceTransformer("all-MiniLM-L6-v2")
        question_embedding = model.encode([question])
        D, I = index.search(question_embedding.astype(np.float32), k=1)
        matched_idx = int(I[0][0])
        print("\n最相关文档内容：\n")
        print(docs[matched_idx])

        # 获取 query + database 的联合 t-SNE 映射，用于 query 点定位
        joint_embed = np.vstack([vectors_array, question_embedding])
        joint_tsne = compute_tsne(joint_embed)
        question_coord = joint_tsne[-1]

        show_tsne_plot(tsne_3d, docs, highlight_idx=matched_idx, question_point=question_coord)
    else:
        show_tsne_plot(tsne_3d, docs)
