from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

import sys
sys.path.append('../')

from qa_chain.get_vectordb import get_vectordb
from qa_chain.model_to_llm import model_to_llm

from langchain_core.messages import SystemMessage


file_path = "../knowledge_db"
persist_path = "../vector_db/chroma"
embedding = "m3e"
top_k = 4


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    vectordb = get_vectordb(file_path, persist_path, embedding)
    retrieved_docs = vectordb.similarity_search(query, k=top_k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


class Chat_QA_chain_self:
    """ "
    使用LangGraph构建的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature
        self.llm = model_to_llm(model, temperature)
        
        # 初始化图
        self._build_graph()

    def _build_graph(self):
        """构建LangGraph图结构"""
        graph_builder = StateGraph(MessagesState)
        
        # 添加节点
        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", ToolNode([retrieve]))
        graph_builder.add_node("generate", self.generate)
        
        # 设置入口点
        graph_builder.set_entry_point("query_or_respond")
        
        # 添加条件边
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        
        # 添加其他边
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
            
        # 编译图
        self.graph = graph_builder.compile(checkpointer=MemorySaver())

    def query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = self.llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate(self, state: MessagesState):
        """Generate answer."""
        # 获取生成的工具消息
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # 格式化提示词
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # 运行
        response = self.llm.invoke(prompt)
        return {"messages": [response]}
    
    def __call__(self, query, thread_id=None):
        """调用图执行问答流程"""
        config = {}
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}
            
        # 执行图并获取结果
        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config
        )
        
        # 返回最后一条消息作为回答
        return result["messages"][-1].content
    
    def stream(self, query, thread_id=None):
        """流式返回问答结果"""
        config = {}
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}
            
        # 返回流式结果
        return self.graph.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
            config=config
        )
