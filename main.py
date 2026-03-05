import streamlit as st
from utils import qa_agent
from langchain.memory import ConversationBufferMemory

st.title("📃 可乐的智能PDF问答工具")

# 侧边栏交互：放置用户的密钥
with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/api-key)")

# 判断会话状态中无外部传入的记忆时，初始化记忆
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

# 上传本地文件的交互
uploaded_file = st.file_uploader("⬆️ 上传PDF文件", type="pdf")
# 用户提问的交互，前提用户已经上传了PDF文件
question = st.text_input("💬 针对PDF文件内容的提问", disabled=not uploaded_file)

# 判断用户是否已输入了openai_api_key,否则返回提示
if uploaded_file and question and not openai_api_key:
    st.info("请输入OpenAI API密钥")

# 是则调用qa_agent函数,返回历史消息列表和AI的回答
if uploaded_file and question and openai_api_key:
    with st.spinner("🤖️AI正在思考中>>>"):
        response = qa_agent(openai_api_key, st.session_state["memory"],
                            uploaded_file, question)
    # AI的回答返回交互
    st.write("### AI回答")
    st.write(response["answer"])

    # 返回历史消息列表并储存在会话状态中
    st.session_state["chat_history"] = response["chat_history"]

# 历史消息列表交互：依次循环当前会话的轮数，每2条会话为一轮，
# 返回每轮会话的头一条为用户消息，下一条为AI消息
# 当消息条数大于2（从0开始计数），本轮会话下方追加一条分割线
if "chat_history" in st.session_state:
    with st.expander("🕐 历史消息列表"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"])-2:
                st.divider()
