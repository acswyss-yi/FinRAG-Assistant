import os
import tempfile
import streamlit as st
from backend.rag import init_models, build_vector_db, build_rag_chain

# 阿里云百炼 API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-7acaf31f9c714127abd3cda28ad8c14e"
API_KEY = os.environ["DASHSCOPE_API_KEY"]


@st.cache_resource
def get_models():
    return init_models(API_KEY)


@st.cache_resource
def get_vector_db(file_path):
    embeddings, _ = get_models()
    return build_vector_db(file_path, embeddings)


def main():
    st.title("FinRAG-Assistant 内部金融财报解读助手")
    st.caption("本地私有知识库，不接入互联网搜索。")

    embeddings, llm = get_models()

    vectorstore = None
    with st.sidebar:
        uploaded_file = st.file_uploader("上传金融PDF文件", type="pdf")
        if uploaded_file:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            st.info("正在构建知识库，请稍候...")
            vectorstore = get_vector_db(file_path)
            st.success("知识库构建完成！")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_text := st.chat_input("请输入您的问题，例如：Q3的营收是多少？"):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        if vectorstore:
            with st.chat_message("assistant"):
                rag_chain = build_rag_chain(vectorstore, llm)
                response = rag_chain.invoke(prompt_text)
                answer = response["answer"]

                st.markdown(answer)

                with st.expander("展开查看检索到的原文片段及来源"):
                    for i, doc in enumerate(response["context"]):
                        source = doc.metadata.get('source', '未知文件')
                        page = doc.metadata.get('page', 0) + 1
                        st.write(f"**片段 {i + 1}** (来源: `{os.path.basename(source)}`, 第 `{page}` 页):")
                        st.write(doc.page_content)

            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("请先在左侧上传文档！")


main()