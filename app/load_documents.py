# Use a more refined context interception method to improve the quality of the FAISS database
import os
import pickle
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def classify_doc_type(text: str) -> str:
    if re.search(r"\d+\.\s+.*\?", text[:300]):
        return "qa"
    elif re.search(r"Article\s+[IVX]+", text[:500]) and re.search(r"Section", text[:1000]):
        return "constitution"
    elif re.search(r"Table of Contents", text[:1000], re.IGNORECASE):
        return "legal_outline"
    else:
        return "policy"


def smart_chunk_documents(documents, max_tokens=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    smart_chunks = []
    for doc in documents:
        doc_type = classify_doc_type(doc.page_content)
        base_meta = doc.metadata.copy()
        base_meta["doc_type"] = doc_type

        if doc_type == "qa":
            qas = re.findall(r"(\d+\.\s+.*?\?)\s+(.*?)(?=\n\d+\.\s+|\Z)", doc.page_content, re.DOTALL)
            for q, a in qas:
                content = f"Q: {q.strip()}\nA: {a.strip()}"
                smart_chunks.append(Document(page_content=content, metadata=base_meta))

        elif doc_type == "constitution":
            parts = re.split(r"(Article\s+[IVX]+[\s\S]*?)(?=Article\s+[IVX]+|\Z)", doc.page_content)
            for part in parts:
                if part.strip():
                    smart_chunks.append(Document(page_content=part.strip(), metadata=base_meta))

        elif doc_type == "legal_outline":
            sections = re.split(r"(?=^[A-Z]{1,2}\.|^[IVX]+\.|^\d+\.\s)", doc.page_content, flags=re.MULTILINE)
            for sec in sections:
                if len(sec.strip()) > 50:
                    smart_chunks.extend(text_splitter.create_documents([sec.strip()], metadatas=[base_meta]))

        else:  # policy: normal paragraphs
            smart_chunks.extend(text_splitter.split_documents([Document(page_content=doc.page_content, metadata=base_meta)]))

    return smart_chunks


def load_all_documents(folder_path="data/pdf_docs"):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
    return all_docs


def embed_documents():
    raw_documents = load_all_documents()
    smart_chunks = smart_chunk_documents(raw_documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(smart_chunks, embeddings)
    vectorstore.save_local("data/vector_index")
    with open("data/vector_index/docs.pkl", "wb") as f:
        pickle.dump(smart_chunks, f)


if __name__ == "__main__":
    embed_documents()
