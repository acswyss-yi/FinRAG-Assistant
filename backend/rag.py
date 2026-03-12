import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

system_prompt = """你是一个专业的金融知识助手。
请仅根据提供的【检索上下文】来回答用户的问题。严禁编造或使用外部知识。
如果上下文中找不到答案，请直接回答"根据提供的文档，无法回答该问题。"。

最重要的一点：你必须在回答的每一处关键数据或结论后，标注信息的来源（文件名和页码）。
【检索上下文】中每一段信息前面都包含了元数据，请严格按照格式引用，例如："根据文档(《2023财报.pdf》, 第5页)显示，营收增长了20%。"

【检索上下文】：
{context}"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


def init_models(api_key: str):
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=api_key
    )
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        model="qwen-turbo",
        temperature=0.1
    )
    return embeddings, llm


def build_vector_db(file_path: str, embeddings, original_name: str = None):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    if original_name:
        for doc in splits:
            doc.metadata["source"] = original_name
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    return vectorstore


def format_context(docs):
    parts = []
    for doc in docs:
        src = os.path.basename(doc.metadata.get('source', '未知文件'))
        page = doc.metadata.get('page', 0) + 1
        parts.append(f"[来源: 《{src}》, 第{page}页]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RunnableParallel(
        context=retriever,
        input=RunnablePassthrough()
    ).assign(
        answer=(
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"context": format_context(x["context"]), "input": x["input"]})
            | chat_prompt
            | llm
            | StrOutputParser()
        )
    )