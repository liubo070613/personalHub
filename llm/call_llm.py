from dotenv import load_dotenv, find_dotenv
from langchain_core.utils import get_from_dict_or_env
import os
def parse_llm_api_key(model:str, env_file:dict()=None): # type: ignore
    """
    通过 model 和 env_file 的来解析平台参数
    """   
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "openai":
        return env_file["OPENAI_API_KEY"]
    elif model == "zhipuai":
        return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
    else:
        raise ValueError(f"model{model} not support!!!")
