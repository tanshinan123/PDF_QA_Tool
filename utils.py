from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


# 定义智能体函数，参数含大模型，记忆体，知识库（上传的文件）及用户会话
def qa_agent(openai_api_key, memory, uploaded_file, question):
    # 选用大语言模型
    model = ChatOpenAI(model="gpt-4-turbo",
                       openai_api_key=openai_api_key,
                       base_url="https://api.aigc369.com/v1")

    # PyPDFLoader会加载本地文件
    # 将存储在内存中的文件内容写入本地文件，返回本地文件存储的路径，加载器加载文件路径
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    # 分割文本块，并限制文本块的字符长度，上下文本块重叠字符长度及分割符号
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", "、", "！", "？", ""]
    )
    texts = text_splitter.split_documents(docs)
    # 嵌入向量模型
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key,
        base_url="https://api.aigc369.com/v1",
        dimensions=1024
    )

    # 文本转化为向量，并存储在FAISS的向量数据库
    vector_database = FAISS.from_documents(texts, embeddings_model)
    # 生成检索器，检索向量数据库的数据
    retriever = vector_database.as_retriever()

    # 组装带记忆的索引增强生成对话链，传入模型，记忆，检索器，
    # 设置外部文档传入模型的方式为映射归约，返回查询的源数据。
    talk = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
    )

    # 使用chat_history键访问和更新对话历史，question赋值为用户查询的字符串
    response = talk.invoke({
        "chat_history": memory,
        "question": question
    })
    return response
