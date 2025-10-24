import streamlit as st
import time
import os
from pathlib import Path  # 用于处理文件路径
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
# 尝试加载dotenv，如果失败则跳过
try:
    from dotenv import load_dotenv
    # 加载.env文件中的环境变量（强制指定路径，确保能找到文件）
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    # 如果没有安装python-dotenv，则跳过
    pass

# 从环境变量中获取配置
os.environ["no_proxy"] = "localhost,127.0.0.1,192.168.1.*"
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))

# 关键修复：将API_KEY注入到OPENAI_API_KEY环境变量（ChatOpenAI默认读取）
if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY

# 页面配置
st.set_page_config(
    page_title="杨幂AI聊天助手",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 隐藏侧边栏按钮
st.markdown("""
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    .stChatMessage {
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        padding: 0.75rem;
    }
    .stUser {
        background-color: #e6f7ff;
    }
    .stAssistant {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# 标题、描述和控制区
col1, col2 = st.columns([3, 1])
with col1:
    st.title("💬 杨幂AI聊天助手")
with col2:
    if st.button("清空对话", type="primary"):
        if "messages" in st.session_state:
            del st.session_state.messages
        st.success("对话已清空")
        st.rerun()

st.caption("Powered by Streamlit & DeepSeek LLM —— 和虚拟形象杨幂对话")
st.divider()

# 调试信息区域（侧边栏）
with st.sidebar:
    st.title("配置调试")
    st.write(f"DEEPSEEK_API_KEY 是否存在：{'✅ 存在' if API_KEY else '❌ 不存在'}")
    st.write(f"API_KEY 前5位：{API_KEY[:5] if API_KEY else '无'}")
    st.write(f"BASE_URL：{BASE_URL if BASE_URL else '❌ 未读取到'}")
    st.write(f"MODEL_NAME：{MODEL_NAME}")
    st.write(f"环境变量 OPENAI_API_KEY 是否存在：{'✅ 存在' if os.getenv('OPENAI_API_KEY') else '❌ 不存在'}")

# 初始化消息列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化DeepSeek LLM函数（修复版）
def init_llm():
    try:
        return ChatOpenAI(
            model_name=MODEL_NAME,
            base_url=BASE_URL,
            temperature=TEMPERATURE,
            timeout=30
            # 无需显式传递api_key，已通过环境变量OPENAI_API_KEY注入
        )
    except Exception as e:
        st.error(f"初始化DeepSeek模型失败: {str(e)}")
        return None

# 消息渲染函数
def render_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

# 系统提示 - 杨幂风格定义
system_prompt = """你是一个智能对话助手，需要模仿明星杨幂的语气和风格回答问题。杨幂的说话风格特点：

一、高情商“金句王”：幽默化解尴尬，常用自嘲、谐音、逻辑跳跃等方式四两拨千斤。
二、北京大妞“嘴毒”：京腔+快语速，爱接话、爱损人，但分寸感强，往往先拿自己开刀。
三、“人间清醒”：输出简短鸡汤，立场明确，爱用短句、对比句，语气淡但观点犀利。
四、直播里的“反流程”：拒绝套路，随时拆台，把“不配合”做成节目效果。
五、声音与语气：带点鼻音，语速快，京腔儿化音，有反差萌。

注意：1. 只输出文本内容，不要包含动作或表情描述；2. 不要提及上述风格特征词。
"""

# 渲染历史消息
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# 输入框处理
if prompt := st.chat_input("输入消息..."):
    # 1. 加入用户消息并渲染
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    # 初始化LLM
    llm = init_llm()
    if not llm:
        st.error("无法初始化DeepSeek语言模型，请检查API设置")
    else:
        # 2. 构建消息列表
        messages = [SystemMessage(content=system_prompt)]
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # 3. 调用API获取回复
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            placeholder.markdown("思考中...")

            try:
                response = llm.invoke(messages)
                full_response = response.content

                # 逐字打印效果
                placeholder.empty()
                display_text = ""
                for sentence in full_response.split('. '):
                    for ch in sentence:
                        display_text += ch
                        placeholder.markdown(display_text)
                        time.sleep(0.01)
                    display_text += '. '
                    placeholder.markdown(display_text)
                    time.sleep(0.1)
            except Exception as e:
                error_msg = f"哎呀，出错了: {str(e)}"
                placeholder.markdown(error_msg)
                full_response = error_msg

        # 4. 加入助手消息
        st.session_state.messages.append({"role": "assistant", "content": full_response})
