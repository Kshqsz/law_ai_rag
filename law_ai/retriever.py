# coding: utf-8
import os
from typing import List, Optional

from langchain.schema.vectorstore import VectorStore
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field, BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from .prompt import MULTI_QUERY_PROMPT_TEMPLATE
from .utils import get_model
from .logger import retriever_logger


class ProxyDuckDuckGoSearch:
    """支持代理的 DuckDuckGo 搜索"""
    
    def __init__(self, proxy: Optional[str] = None, timeout: int = 15):
        self.proxy = proxy or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        self.timeout = timeout
    
    def results(self, query: str, max_results: int = 2) -> List[dict]:
        import time
        import random
        
        # 尝试 html 和 lite 后端（api 容易限速）
        backends = ["html", "lite"]
        
        for backend in backends:
            try:
                delay = random.uniform(2, 4)
                retriever_logger.debug(f"  🦆 DuckDuckGo ({backend}) 等待 {delay:.1f}s...")
                time.sleep(delay)
                
                with DDGS(proxies=self.proxy, timeout=self.timeout) as ddgs:
                    results = list(ddgs.text(query, max_results=max_results, backend=backend))
                    formatted = []
                    for r in results:
                        formatted.append({
                            "title": r.get("title", ""),
                            "link": r.get("href", ""),
                            "snippet": r.get("body", "")
                        })
                    if formatted:
                        retriever_logger.info(f"✓ DuckDuckGo ({backend}) 成功找到 {len(formatted)} 条结果")
                        return formatted
            except Exception as e:
                retriever_logger.debug(f"  ✗ {backend} 失败: {str(e)[:50]}")
                continue
        
        retriever_logger.warning(f"⚠ 网页搜索失败: 所有后端均无可用结果")
        return []


class LawWebRetiever(BaseRetriever):
    # Inputs
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )

    search: ProxyDuckDuckGoSearch = Field(..., description="DuckDuckGo Search with Proxy")
    num_search_results: int = Field(1, description="Number of pages per search")

    text_splitter: TextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )
    
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        retriever_logger.info(f"🔍 开始网页搜索: '{query}'")

        results = self.search.results(query, self.num_search_results)
        if results:
            retriever_logger.info(f"✓ 网页搜索完成，找到 {len(results)} 条结果")
            for i, res in enumerate(results, 1):
                retriever_logger.info(f"  📄 网页{i}: {res.get('title', 'N/A')[:50]}...")

        docs = []
        for res in results:
            docs.append(Document(
                page_content=res["snippet"],
                metadata={"link": res["link"], "title": res["title"]}
            ))

        docs = self.text_splitter.split_documents(docs)

        return docs


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


def get_multi_query_law_retiever(retriever: BaseRetriever, model: BaseModel) -> BaseRetriever:
    output_parser = LineListOutputParser()

    llm_chain = LLMChain(llm=model, prompt=MULTI_QUERY_PROMPT_TEMPLATE, output_parser=output_parser)

    retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
    )

    return retriever
