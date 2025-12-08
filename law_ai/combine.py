# coding: utf-8
from typing import List
from collections import defaultdict

from langchain.docstore.document import Document

# example:
#   相关法律：《中华人民共和国刑法》
#   第二百三十二条 故意杀人的，处死刑。
#   第二百三十三条 过失致人死亡的，处三年以上七年以下有期徒刑
def combine_law_docs(docs: List[Document]) -> str:
    law_books = defaultdict(list)
    for doc in docs:
        metadata = doc.metadata
        if 'book' in metadata:
            law_books[metadata["book"]].append(doc)

    law_str = ""
    for book, docs in law_books.items():
        law_str += f"相关法律：《{book}》\n"
        law_str += "\n".join([doc.page_content.strip("\n") for doc in docs])
        law_str += "\n"

    return law_str


def combine_web_docs(docs: List[Document]) -> str:
    web_str = ""
    for doc in docs:
        web_str += f"相关网页：{doc.metadata['title']}\n"
        web_str += f"网页地址：{doc.metadata['link']}\n"
        web_str += doc.page_content.strip("\n") + "\n"
        web_str += "\n"

    return web_str
