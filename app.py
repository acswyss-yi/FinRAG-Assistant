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


def apply_styles():
    st.markdown("""
    <style>
    /* 侧边栏整体 */
    [data-testid="stSidebar"] {
        background-color: #F0F2F6;
        border-right: 1px solid #E0E3EA;
        padding: 1.5rem 1rem;
    }
    /* 上传拖拽区域外框 */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #fff !important;
        border: 1.5px dashed #C0C8D8 !important;
        border-radius: 12px !important;
        padding: 1.8rem 1rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 0.4rem !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
        transition: border-color 0.2s, background 0.2s !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4A7FC1 !important;
        background: #F5F8FF !important;
    }
    /* 上传图标 */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
        width: 52px !important;
        height: 52px !important;
        color: #8A9BB5 !important;
        margin-bottom: 0.3rem !important;
    }
    /* 浏览文件按钮 */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: #2C3E55 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.45rem 1.4rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        margin-top: 0.6rem !important;
        cursor: pointer !important;
        transition: background 0.2s !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
        background: #1A2B3C !important;
    }
    /* 隐藏右上角 Deploy 按钮 */
    [data-testid="stToolbar"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    apply_styles()
    st.title("FinRAG-Assistant 内部金融财报解读助手")
    st.caption("本地私有知识库，不接入互联网搜索。")

    embeddings, llm = get_models()

    vectorstore = None
    with st.sidebar:
        st.markdown("""
        <div style="font-size:1.2rem; font-weight:700; color:#1A2332;
                    padding-bottom:0.6rem; margin-top:-1rem; margin-bottom:1.6rem;
                    border-bottom:2px solid #4A7FC1;">
            上传金融财报PDF文件
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type="pdf")
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