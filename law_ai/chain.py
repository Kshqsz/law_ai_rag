# coding: utf-8
from typing import Any, Optional, List
from collections import defaultdict
from operator import itemgetter

from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.schema import format_document
from langchain.schema import BaseRetriever
from langchain.pydantic_v1 import Field
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import BooleanOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.chains.base import Chain
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from .utils import get_vectorstore, get_model
from .retriever import LawWebRetiever, ProxyDuckDuckGoSearch, get_multi_query_law_retiever
from .prompt import LAW_PROMPT, CHECK_LAW_PROMPT, HYPO_QUESTION_PROMPT
from .combine import combine_law_docs, combine_web_docs
from .logger import chain_logger


class LawStuffDocumentsChain(StuffDocumentsChain):
    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:
        # Join the documents together to put them in the prompt.
        law_book = defaultdict(list)
        law_web = defaultdict(list)
        for doc in docs:
            metadata = doc.metadata
            if 'book' in metadata:
                law_book[metadata["book"]].append(
                    format_document(doc, self.document_prompt).strip("\n"))
            elif 'link' in metadata:
                law_web[metadata["title"]].append(
                    format_document(doc, self.document_prompt).strip("\n"))

        law_str = ""
        for book, page_contents in law_book.items():
            law_str += f"《{book}》\n"
            law_str += "\n".join(page_contents)
            law_str += "\n\n"

        for web, page_contents in law_web.items():
            law_str += f"网页：{web}\n"
            law_str += "\n".join(page_contents)
            law_str += "\n\n"

        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = law_str
        return inputs


class LawQAChain(BaseRetrievalQA):
    vs_retriever: BaseRetriever = Field(exclude=True)
    web_retriever: BaseRetriever = Field(exclude=True)

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        chain_logger.info(f"📚 开始检索法律文献...")
        vs_docs = self.vs_retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        chain_logger.info(f"✓ 法律文献检索完成，找到 {len(vs_docs)} 条相关文档")

        chain_logger.info(f"🌐 开始检索网页资源...")
        web_docs = self.web_retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        chain_logger.info(f"✓ 网页资源检索完成，找到 {len(web_docs)} 条相关资源")

        total_docs = vs_docs + web_docs
        chain_logger.info(f"📖 共检索到 {len(total_docs)} 条资料")
        return total_docs

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        chain_logger.info(f"📚 开始检索法律文献...")
        vs_docs = await self.vs_retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        chain_logger.info(f"✓ 法律文献检索完成，找到 {len(vs_docs)} 条相关文档")

        chain_logger.info(f"🌐 开始检索网页资源...")
        web_docs = await self.web_retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        chain_logger.info(f"✓ 网页资源检索完成，找到 {len(web_docs)} 条相关资源")

        total_docs = vs_docs + web_docs
        chain_logger.info(f"📖 共检索到 {len(total_docs)} 条资料")
        return total_docs

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "law_qa"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt, callbacks=callbacks)
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        combine_documents_chain = LawStuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=callbacks,
        )

        return cls(
            combine_documents_chain=combine_documents_chain,
            callbacks=callbacks,
            **kwargs,
        )


def get_check_law_chain(config: Any) -> Chain:
    model = get_model()

    check_chain = CHECK_LAW_PROMPT | model | BooleanOutputParser()

    return check_chain


def get_law_chain(config: Any, out_callback: AsyncIteratorCallbackHandler) -> Chain:
    chain_logger.info("🔧 初始化法律 RAG Chain...")
    
    law_vs = get_vectorstore(config.LAW_VS_COLLECTION_NAME)
    web_vs = get_vectorstore(config.WEB_VS_COLLECTION_NAME)
    chain_logger.info("✓ 向量库加载完成")

    vs_retriever = law_vs.as_retriever(search_kwargs={"k": config.LAW_VS_SEARCH_K})
    
    # 使用代理配置
    proxy = getattr(config, 'WEB_PROXY', None)
    web_retriever = LawWebRetiever(
        vectorstore=web_vs,
        search=ProxyDuckDuckGoSearch(proxy=proxy),
        num_search_results=config.WEB_VS_SEARCH_K
    )
    chain_logger.info(f"✓ 检索器初始化完成 (代理: {proxy or '无'})")

    multi_query_retriver = get_multi_query_law_retiever(vs_retriever, get_model())
    chain_logger.info("✓ 多查询检索器初始化完成")

    callbacks = [out_callback] if out_callback else []

    def log_law_docs(x):
        """记录法律文档检索结果"""
        law_docs = x["law_docs"]
        chain_logger.info(f"📚 向量库检索完成，找到 {len(law_docs)} 条法律文档:")
        for i, doc in enumerate(law_docs, 1):
            book = doc.metadata.get('book', '未知')
            content_preview = doc.page_content[:80].replace('\n', ' ')
            chain_logger.info(f"  📖 [{i}] 《{book}》: {content_preview}...")
        return law_docs
    
    def log_web_docs(x):
        """记录网页检索结果"""
        web_docs = x["web_docs"]
        if web_docs:
            chain_logger.info(f"🌐 网页检索完成，找到 {len(web_docs)} 条网页资源")
        return web_docs
    
    def log_prompt_and_call_llm(x):
        """记录 prompt 并调用大模型"""
        law_context = x["law_context"]
        web_context = x["web_context"]
        question = x["question"]
        
        chain_logger.info("=" * 60)
        chain_logger.info("📝 发送给大模型的 Prompt:")
        chain_logger.info(f"  用户问题: {question}")
        chain_logger.info(f"  法律上下文 ({len(law_context)} 字符):")
        # 只显示前 500 字符
        preview = law_context[:500].replace('\n', '\n    ')
        chain_logger.info(f"    {preview}...")
        if web_context:
            chain_logger.info(f"  网页上下文 ({len(web_context)} 字符)")
        chain_logger.info("=" * 60)
        chain_logger.info("🤖 调用大模型生成答案...")
        
        # 格式化 prompt 并调用模型
        prompt = LAW_PROMPT
        answer_chain = prompt | get_model(callbacks=callbacks) | StrOutputParser()
        return answer_chain.invoke(x)

    chain = (
        RunnableMap(
            {
                "law_docs": itemgetter("question") | multi_query_retriver,
                'web_docs': itemgetter("question") | web_retriever,
                "question": lambda x: x["question"]}
        )
        | RunnableMap(
            {
                "law_docs": log_law_docs,
                "web_docs": log_web_docs,
                "law_context": lambda x: combine_law_docs(x["law_docs"]),
                "web_context": lambda x: combine_web_docs(x["web_docs"]),
                "question": lambda x: x["question"]}
        )
        | RunnableMap({
            "law_docs": lambda x: x["law_docs"],
            "web_docs": lambda x: x["web_docs"],
            "law_context": lambda x: x["law_context"],
            "web_context": lambda x: x["web_context"],
            "question": lambda x: x["question"],
            "answer": log_prompt_and_call_llm
        })
    )

    return chain


def get_hypo_questions_chain(config: Any) -> Chain:
    model = get_model()

    functions = [
        {
            "name": "hypothetical_questions",
            "description": "Generate hypothetical questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    },
                },
                "required": ["questions"]
            }
        }
    ]

    chain = (
        {"context": lambda x: f"《{x.metadata['book']}》{x.page_content}"}
        | HYPO_QUESTION_PROMPT
        | model.bind(functions=functions, function_call={"name": "hypothetical_questions"})
        | JsonKeyOutputFunctionsParser(key_name="questions")
    )

    return chain