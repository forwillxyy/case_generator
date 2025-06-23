import os
from langchain_openai import ChatOpenAI
from utils.csv_handler import *


def get_model():
    LLM_URL = "https://gptapi.com/v1"
    LLM_MODEL = "gpt-4o"
    LLM_API_KEY = os.getenv("GPT_TOKEN")
    model = ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_URL)
    return model


# convert GPT output to json list string
def convert_to_json_list_str(input_str):
    left_index = input_str.find("[")
    right_index = input_str.rfind("]")
    format_str = input_str
    if left_index > -1 and right_index > -1:
        format_str = input_str[left_index:right_index + 1]
    return format_str


# convert GPT output to json object string
def convert_to_json_obj_str(input_str):
    left_index = input_str.find("{")
    right_index = input_str.rfind("}")
    format_str = input_str
    if left_index > -1 and right_index > -1:
        format_str = input_str[left_index:right_index + 1]
    return format_str


# save data to file
def save_case_to_file(file_path, data):
    if not data or len(data) == 0:
        print("No data to save")
        return

    headers = ("service_name,service_host,api_name,case_description,case_name,http_method,headers,url,params_body,"
               "validation,case_link,case_link_name,is_create_jira_case").split(",")

    content = [headers]
    for item in data:
        content.append(item)

    csv_writer(file_path, content)
