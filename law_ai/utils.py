# coding: utf-8
import os
from typing import List, Dict
from collections import defaultdict
import dashscope

from langchain.docstore.document import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores import Chroma
from langchain.indexes._api import _batch
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import Callbacks


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope Embedding 模型"""
    
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        # DashScope 单次最多支持 25 条，需要分批处理
        all_embeddings = []
        batch_size = 25

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            response = dashscope.TextEmbedding.call(
                model = self.model,
                input = batch_texts,
                api_key = self.api_key
            )
            if response.status_code == 200:
                batch_embeddings = [item["embedding"] for item in response.output["embeddings"]]
                all_embeddings.extend(batch_embeddings)
            else:
                raise Exception(f"DashScope Embedding 调用失败: {response.code} - {response.message}")
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """单条文本向量化"""
        return self.embed_documents([text])[0]


def get_embedding_model() -> DashScopeEmbeddings:
    """获取 Embedding 模型，使用 DashScope 原生 API"""
    return DashScopeEmbeddings()


def get_cached_embedder() -> CacheBackedEmbeddings:
    fs = LocalFileStore("./.cached/embeddings")
    underlying_embeddings = get_embedding_model()

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )
    return cached_embedder

def get_record_manager(namespace: str = "law") -> SQLRecordManager:
    return SQLRecordManager(
        f"chroma/{namespace}", db_url="sqlite:///law_record_manager_cache.sql"
    )


def get_vectorstore(collection_name: str = "law") -> Chroma:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_cached_embedder(),
        collection_name=collection_name
    )
    return vectorstore


def clear_vectorstore(collection_name: str = "law") -> None:
    record_manager = get_record_manager(collection_name)
    vectorstore = get_vectorstore(collection_name)

    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")


def get_model(
        model: str = None,
        streaming: bool = True,
        callbacks: Callbacks = None) -> ChatOpenAI:
    """获取 LLM 模型，支持 DashScope API"""
    model_name = model or os.getenv("MODEL_NAME", "qwen-plus")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    
    return ChatOpenAI(
        model=model_name,
        streaming=streaming,
        callbacks=callbacks,
        openai_api_key=api_key,
        openai_api_base=base_url
    )


def law_index(docs: List[Document], show_progress: bool = True) -> Dict:
    info = defaultdict(int)

    record_manager = get_record_manager("law")
    vectorstore = get_vectorstore("law")

    pbar = None
    if (show_progress):
        from tqdm import tqdm
        pbar = tqdm(total=len(docs))
    
    for batch in _batch(100, docs):
        result = index(
            batch,
            record_manager,
            vectorstore,
            cleanup=None,
            # cleanup="full"
            source_id_key="source"
        )
        for k, v in result.item():
            info[k] += v
        if pbar:
            pbar.update(len(docs))
    if pbar:
        pbar.close()
    return dict(info)