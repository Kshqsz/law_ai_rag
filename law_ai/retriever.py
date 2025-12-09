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

import time
import random
import requests


class ProxyDuckDuckGoSearch:
    """支持代理的 DuckDuckGo 搜索 - 带 Rate Limit 处理和 Bing 备选"""
    
    def __init__(self, proxy: Optional[str] = None, timeout: int = 30):
        self.proxy = proxy or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        self.timeout = timeout
        self.proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        }
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[dict]:
        """尝试 DuckDuckGo 搜索"""
        backends = ["api", "html", "lite"]
        
        for backend in backends:
            try:
                delay = random.uniform(3, 6)
                retriever_logger.info(f"🦆 DuckDuckGo ({backend}) 等待 {delay:.1f}s...")
                time.sleep(delay)
                
                with DDGS(proxies=self.proxy, timeout=self.timeout) as ddgs:
                    results = list(ddgs.text(query, max_results=max_results, backend=backend))
                    if results:
                        return [{"title": r.get("title", ""), "link": r.get("href", ""), 
                                "snippet": r.get("body", "")} for r in results]
            except Exception as e:
                if "Ratelimit" in str(e):
                    continue
                retriever_logger.debug(f"  {backend} 失败: {str(e)[:40]}")
        return []
    
    def _search_bing(self, query: str, max_results: int) -> List[dict]:
        """使用 Bing 搜索（爬取方式）"""
        try:
            import re
            from html import unescape
            
            retriever_logger.info("🔎 尝试 Bing 搜索...")
            url = "https://www.bing.com/search"
            params = {"q": query, "count": max_results * 2}
            
            resp = requests.get(url, params=params, headers=self._headers, 
                              proxies=self.proxies, timeout=15)
            
            if resp.status_code != 200:
                retriever_logger.debug(f"  Bing 返回 {resp.status_code}")
                return []
            
            html = resp.text
            results = []
            
            # 提取搜索结果 (简化的正则匹配)
            pattern = r'<li class="b_algo"[^>]*>.*?<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>.*?<p[^>]*>(.*?)</p>'
            matches = re.findall(pattern, html, re.DOTALL)
            
            for link, title, snippet in matches[:max_results]:
                # 清理 HTML
                title = re.sub(r'<[^>]+>', '', title)
                snippet = re.sub(r'<[^>]+>', '', snippet)
                results.append({
                    "title": unescape(title.strip()),
                    "link": link,
                    "snippet": unescape(snippet.strip())[:200]
                })
            
            if results:
                retriever_logger.info(f"✓ Bing 搜索成功，找到 {len(results)} 条结果")
            return results
        except Exception as e:
            retriever_logger.debug(f"  Bing 失败: {str(e)[:50]}")
            return []
    
    def _search_google(self, query: str, max_results: int) -> List[dict]:
        """使用 Google 搜索（爬取方式）- 最后备选"""
        try:
            import re
            from html import unescape
            from urllib.parse import unquote
            
            retriever_logger.info("🔎 尝试 Google 搜索...")
            url = "https://www.google.com/search"
            params = {"q": query, "num": max_results * 2}
            
            resp = requests.get(url, params=params, headers=self._headers,
                              proxies=self.proxies, timeout=15)
            
            if resp.status_code != 200:
                return []
            
            html = resp.text
            results = []
            
            # 提取搜索结果
            link_pattern = r'/url\?q=([^&]+)&'
            links = re.findall(link_pattern, html)
            
            for link in links[:max_results]:
                link = unquote(link)
                if link.startswith('http') and 'google.com' not in link:
                    results.append({
                        "title": link.split('/')[2] if '/' in link else link,
                        "link": link,
                        "snippet": ""
                    })
            
            if results:
                retriever_logger.info(f"✓ Google 搜索成功，找到 {len(results)} 条结果")
            return results
        except Exception as e:
            retriever_logger.debug(f"  Google 失败: {str(e)[:50]}")
            return []
    
    def results(self, query: str, max_results: int = 2) -> List[dict]:
        """
        执行搜索：DuckDuckGo -> Bing -> Google
        """
        # 1. 先尝试 DuckDuckGo
        results = self._search_duckduckgo(query, max_results)
        if results:
            retriever_logger.info(f"✓ DuckDuckGo 搜索成功")
            return results
        
        # 2. DuckDuckGo 失败，尝试 Bing
        retriever_logger.warning("⚠ DuckDuckGo 被限速，尝试备用搜索引擎...")
        results = self._search_bing(query, max_results)
        if results:
            return results
        
        # 3. Bing 失败，尝试 Google
        results = self._search_google(query, max_results)
        if results:
            return results
        
        retriever_logger.warning("⚠ 所有搜索引擎均失败，跳过网页搜索")
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
