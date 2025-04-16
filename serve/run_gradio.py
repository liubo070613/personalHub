# 导入必要的库

import sys
import os                # 用于操作系统相关的操作，例如读取环境变量

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import IPython.display   # 用于在 IPython 环境中显示数据，例如图片
import io                # 用于处理流式数据（例如文件流）
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from qa_chain.chain import Chat_QA_chain_self
from fastapi import FastAPI
import uuid
# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量
_ = load_dotenv(find_dotenv())
LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    "zhipuai": ["glm-4-long"]
}


LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
INIT_LLM = "glm-4-long"
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = "../knowledge_db"
DEFAULT_PERSIST_PATH = "../vector_db/chroma"
AIGC_AVATAR_PATH = "../figures/aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "../figures/datawhale_avatar.png"
AIGC_LOGO_PATH = "../figures/aigc_logo.png"
DATAWHALE_LOGO_PATH = "../figures/datawhale_logo.png"

def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")


def handle_chatbot(query, chatbot_history, llm):
    try:
        qa_chain = Chat_QA_chain_self(llm)
        ai_response = qa_chain(query, 123)
        chatbot_history.append([query, ai_response])
        return "", chatbot_history
    except Exception as e:
        chatbot_history.append([query, f"Error: {str(e)}"])
        return "", chatbot_history


block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):           
        gr.Image(value=AIGC_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)
   
        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>基于知识库的问答系统</center></h1>
                <center>Knowledge Base Assistant</center>
                """)
        gr.Image(value=DATAWHALE_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True, avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH))
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

        with gr.Column(scale=1):
            file = gr.File(label='请选择知识库目录', file_count='directory',
                           file_types=['.txt', '.md', '.docx', '.pdf'])
            with gr.Row():
                init_db = gr.Button("知识库文件向量化")

            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_LLM,
                    interactive=True)

                embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                         label="Embedding model",
                                         value=INIT_EMBEDDING_MODEL)

        # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
        msg.submit(
            handle_chatbot,
            inputs=[msg, chatbot, llm],  
            outputs=[msg, chatbot]
        )
    gr.Markdown("""提醒：<br>
    1. 使用时请先上传自己的知识文件，不然将会解析项目自带的知识库。
    2. 初始化数据库时间可能较长，请耐心等待。
    3. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()

app = FastAPI()

app=gr.mount_gradio_app(app, demo, path="/")

# 在文件中添加流式处理函数
import uuid

# 在应用中创建会话ID字典
session_ids = {}

def process_stream(query, model_name, chat_id=None):
    # 获取或创建会话ID
    if chat_id not in session_ids:
        session_ids[chat_id] = str(uuid.uuid4())
    
    thread_id = session_ids[chat_id]
    
    # 初始化模型
    qa_chain = Chat_QA_chain_self(model_name)
    
    # 清空输入框
    msg_value = ""
    
    # 准备聊天记录
    chatbot_value = chatbot.value.copy()
    chatbot_value.append([query, ""])
    
    # 流式生成回答
    full_response = ""
    for chunk in qa_chain.stream(query, thread_id=thread_id):
        if "messages" in chunk and chunk["messages"]:
            last_message = chunk["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                full_response += last_message.content
                # 更新界面
                chatbot_value[-1][1] = full_response
                yield msg_value, chatbot_value
    
    return msg_value, chatbot_value

