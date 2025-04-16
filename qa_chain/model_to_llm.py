import sys

sys.path.append("../llm")
# from llm.zhipuai_llm import ZhipuAILLM
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
from llm.call_llm import parse_llm_api_key


def model_to_llm(
    model: str = None,
    temperature: float = 0.0,
    api_key: str = None,
):
    """
    星火：model,temperature,appid,api_key,api_secret
    百度问心：model,temperature,api_key,api_secret
    智谱：model,temperature,api_key
    OpenAI：model,temperature,api_key
    """
    if model in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-0613",
        "gpt-4",
        "gpt-4-32k",
    ]:
        if api_key == None:
            api_key = parse_llm_api_key("openai")
        llm = ChatOpenAI(
            model_name=model, temperature=temperature, openai_api_key=api_key
        )
    elif model in ["glm-4-long"]:
        if api_key == None:
            api_key = parse_llm_api_key("zhipuai")
        llm = ChatZhipuAI(model=model, zhipuai_api_key=api_key, temperature=temperature)
    else:
        raise ValueError(f"model:{model} not support!!!")
    return llm
