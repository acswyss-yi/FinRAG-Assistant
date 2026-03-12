# Financial RAG Assistant  金融财报助手

基于 RAG（检索增强生成）的财报问答系统，使用阿里云百炼（通义千问）模型。

**Demo：[http://8.149.245.12:8501/](http://8.149.245.12:8501/)**

## 功能

- 上传金融 PDF 文档，自动构建本地向量知识库
- 基于文档内容回答问题，严禁编造，拒绝使用外部知识
- 每条回答自动标注来源文件名和页码
- 支持展开查看检索到的原文片段

## 技术栈

| 组件 | 说明 |
|------|------|
| [Streamlit](https://streamlit.io/) | Web UI 框架 |
| [LangChain](https://www.langchain.com/) | RAG 链路编排 |
| [Chroma](https://www.trychroma.com/) | 本地向量数据库 |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF 解析 |
| 通义千问 text-embedding-v3 | 文本向量化 |
| 通义千问 qwen-turbo | 大语言模型 |

## 项目结构

```
RAG Knowledge Assistant/
├── app.py              # Streamlit UI
├── backend/
│   ├── __init__.py
│   └── rag.py          # RAG 逻辑（模型、向量库、检索链）
├── requirements.txt
└── chroma_db/          # 向量数据库（运行后自动生成）
```

## 环境要求

- Python 3.10+
- 阿里云百炼 API Key（[控制台获取](https://bailian.console.aliyun.com/)）

## 安装

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```


## 运行

```bash
streamlit run app.py
```

