# coding: utf-8
import os
from typing import List, Dict
from collections import defaultdict

from langchain.docstore.document import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores import Chroma
from langchain.indexes._api import _batch
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import Callbacks


def get_cached_embedder() -> CacheBackedEmbeddings:
    fs = LocalFileStore("./.cache/embeddings")
    base_url = os.getenv("OPENAI_BASE_URL")
    # 阿里云 DashScope 的 embedding 模型
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    
    underlying_embeddings = OpenAIEmbeddings(
        model=embedding_model,
        base_url=base_url
    )

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
        collection_name=collection_name)

    return vectorstore


def clear_vectorstore(collection_name: str = "law") -> None:
    record_manager = get_record_manager(collection_name)
    vectorstore = get_vectorstore(collection_name)

    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")


def get_model(
        model: str = None,
        streaming: bool = True,
        callbacks: Callbacks = None) -> ChatOpenAI:
    # 从环境变量获取模型名称，默认使用 qwen-max
    model_name = model or os.getenv("MODEL_NAME", "qwen-max")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    model = ChatOpenAI(
        model=model_name,
        streaming=streaming,
        callbacks=callbacks,
        base_url=base_url
    )
    return model


def law_index(docs: List[Document], show_progress: bool = True) -> Dict:
    info = defaultdict(int)

    record_manager = get_record_manager("law")
    vectorstore = get_vectorstore("law")

    pbar = None
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(docs))

    for docs in _batch(100, docs):
        result = index(
            docs,
            record_manager,
            vectorstore,
            cleanup=None,
            # cleanup="full",
            source_id_key="source",
        )
        for k, v in result.items():
            info[k] += v

        if pbar:
            pbar.update(len(docs))

    if pbar:
        pbar.close()

    return dict(info)
