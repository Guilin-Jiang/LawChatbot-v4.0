import pickle
from langchain.schema import Document
import re
import spacy
import networkx as nx
import matplotlib.pyplot as plt

# 加载已生成的 chunked 文档
with open("data/vector_index/docs.pkl", "rb") as f:
    documents: list[Document] = pickle.load(f)

# 抽取实体与关系
nlp = spacy.load("en_core_web_sm") #get a llm used for extrating

def extract_entities_and_relations(doc: Document):
    entities = set()
    relations = []

    # 抽取条文号
    article_match = re.search(r"(Article\s+[IVX]+|第[一二三四五六七八九十百]+条)", doc.page_content)
    if article_match:
        entities.add(article_match.group())

    # 用 spaCy 或 HanLP 抽取名词短语
    spacy_doc = nlp(doc.page_content)
    for chunk in spacy_doc.noun_chunks:
        if len(chunk.text.strip()) > 1:
            entities.add(chunk.text.strip())

    # 简化：提取“X 拥有 Y”结构作为关系
    for sent in spacy_doc.sents:
        if "has" in sent.text or "enjoy" in sent.text or "assume" in sent.text:
            relations.append((sent.start_char, sent.text.strip()))

    return list(entities), relations

# 构建图谱结构
if __name__ == '__main__':
    G = nx.DiGraph()

    for doc in documents:
        entities, relations = extract_entities_and_relations(doc)
        doc_id = doc.metadata.get("source", "unknown") + "_" + doc.page_content[:20]

        # 将实体添加为节点
        for ent in entities:
            G.add_node(ent, doc=doc_id)

        # 添加简单的关系（例如 A has B）
        for _, rel_text in relations:
            match = re.match(r"(.*?) (has|enjoy|assume) (.*)", rel_text)
            if match:
                subject, verb, obj = match.groups()
                G.add_edge(subject.strip(), obj.strip(), label=verb.strip(), doc=doc_id)

    # 可视化和保存图谱
    with open("data/know_graph/graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=800, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):d['label'] for u,v,d in G.edges(data=True)})
    plt.savefig("data/know_graph/graph_visual.png", dpi=300, bbox_inches='tight')
    plt.close()
