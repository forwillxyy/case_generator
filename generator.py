import json
import os
import sys

from typing import TypedDict
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_test_code import add_automation_test_case
from tools import *

CASE_COVERAGE_THRESHOLD = 90

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = "mustang_backend"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://langsmith.aaa.net/api"


# structure to store state
class State(TypedDict):
    api_contract: str
    service_host: str
    generated_cases: str
    case_coverage: int
    case_coverage_details: str
    absent_cases: str
    user_input: str
    custom_prompts: str


# class to generate cases and automation code
class LLMWorker:

    model = get_model()

    # expected scenarios for different data types of API fields
    FIELD_EXPECTED_SCENARIOS = {"string": ["valid", "whitespace only", "empty", "special characters", "missing"],
                                "numeric(int/float)": ["valid", "not number", "negative number", "zero", "max value",
                                                       "min value", "missing"],
                                "enum": ["valid", "invalid", "missing"],
                                "boolean": ["true", "false", "missing"],
                                "repeated(list/array)": ["single", "multiple", "empty list", "missing"],
                                "optional": ["valid", "missing"]}

    # API contract
    api_contract = ""
    # service host
    service_host = ""

    def __init__(self, api_contract, service_host, automate_case=False, **kwargs):
        self.api_contract = api_contract
        self.service_host = service_host
        self.automate_case = automate_case
        custom_args = kwargs
        self.custom_prompts = ""
        if custom_args:
            for key, value in custom_args.items():
                self.custom_prompts += f"{key}: {value}\n"

    """
        prompts for case generation.
        1. first to generate cases based on API contract
        2. if coverage is not enough, check coverage and generate more cases
    """
    case_generator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a QA expert，your task is to design test cases based on following API definition.
                   Your test cases should cover as many scenarios as possible for all fields, including nested fields.
                   Please design as many param validation cases as possible for all fields, including nested fields.
                   Please output full test cases in one answer, thanks.
                   Important: if you are asked to design more cases, please review the existing cases and 
                   case check result, then generate missing cases based on check result, 
                   but do NOT generate existing cases again. 
                   Please remove duplicate cases if exist, thanks."""),
            (
                "system",
                """For each field, including nested fields, please design cases based on param data type. 
                   To generate expected cases of different data types, please refer to: {expected_scenarios},
                   please consider all scenarios for each field, including nested fields.
                   If param is string, please only refer to expected scenarios for string type, 
                   if param is numeric, please only refer to expected scenarios for numeric type... 
                   For example: if there are 3 params in API contract, param1 and param2 are string, param3 is numeric,
                   then there should be 5 expected cases for param1 and param2, including ["valid", "whitespace only", 
                   "empty", "special characters", "missing"]; 7 expected cases for param3, including ["valid", 
                   "not number", "negative number", "zero", "max value", "min value","missing"], thanks."""),
            (
                "human",
                "{api_contract}"),
            (
                "human",
                """Service host: {service_host}; service_name should be lower case,
                   and - between words; api_name should be lower case, and _ between words"""),
            (
                "human",
                """Please consider following knowledge {custom_prompts}"""
            ),
            (
                "human",
                """Please output cases as a json array following below format：
                   [{{"service_name": "test-gpt", "api_name": "update_fleet", "http_method": "GET",
                   "service_host": "http://myservice.com/view/service",
                   "url": "/kartaview_service/v1/videos",
                   "json": {{"video_id": 123, operations: [{{"device_id": "A123", "operation_type": "BIND"}}]}},
                   "validation": {{"status_code": 200}}, "case_name": "get_video_by_correct_id"}}].
                   please follow this format strictly, make sure each case can be parsed as a json object.
                   Please convert output as one line string to output, no unnecessary whitespaces,
                   no new line characters, no tab characters, thanks.
                   Please just output the cases, no other information, no ` characters."""),
            (
                "human",
                """The value of "case_name" should be lower case, and _ between words, like "get_video";
                   if params are body type, please use "json": {{"video_id": 123, operations: [{{"device_id": "A123", "operation_type": "BIND"}}]}} format in case description,
                   if params are query type or method is GET, please use "params": "video_id=123&page=1" format in case description,
                   if params are header type, please use "headers": {{"Content-Type": "application/json"}} format in case description,
                   if params are form type, please use "json": {{"video_id": 123}} format in case description,
                   if params are json type, please use "json": {{"video_id": 123}} format in case description;"
                   if one param is path type, please do NOT add it to "params_body", just add it to url, like "/videos/123" in case description;"
                   for such query param, "repeated string filters = 1 [(validate.rules).repeated.items.string.pattern = \"^(fleet_id|fleet_name).*$\"]",
                   the generated "params" should be "filters=fleet_id.eq.123&filters=fleet_name.eq.test_fleet",
                   it is better to not include "_service" for the value of "service_name",
                   the value of "service_host" should be "API host" which is defined by user content, thanks."""),
            (
                "human",
                "Existing cases are: {existing_cases}"),
            (
                "human",
                "Case coverage details are: {case_coverage}"),
            MessagesPlaceholder(variable_name="user_input"),
        ],
    )

    """
        prompts for checking coverage. check if the generated test cases have covered expected scenarios.
        1. if coverage is not enough, ask case generator to design missing cases
        2. if coverage is enough, ask code generator to generate automation test code
    """
    check_coverage_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a QA expert, please check if the generated test cases have covered expected scenarios 
                   for each field of API contract, including nested fields. The expected scenarios for different 
                   data types of params can refer to : {expected_scenarios}. 
                   
                   Please output the result following json format: {{"coverage": 85, "details": "ss"}}, 
                   please just output json format string, no other information, no ` characters, 
                   make sure if can be parsed as json object directly.
                   
                   When calculating coverage, please recognize the data type for each param, and refer to expected 
                   scenarios for this type. If param is string, please only refer to expected scenarios for string type, 
                   if param is numeric, please only refer to expected scenarios for numeric type... 
                   For example: if there are 3 params in API contract, param1 and param2 are string, param3 is numeric,
                   then there should be 5 expected cases for param1 and param2, including ["valid", "whitespace only", 
                   "empty", "special characters", "missing"]; 4 expected cases for param3, including ["valid", 
                   "not number", "negative number", "zero", "max value", "min value", "missing"], then total expected 
                   case number of this API should be (2*5)+7=17. 
                   If the generated cases cover ["valid", "whitespace only", "empty", "missing"] for param1, cover 
                   ["valid", "empty", "special characters", "missing"] for param2, and cover ["valid", "not number", 
                   "missing"] for param3, then actual generated case number is 4+4+3=11, 
                   the coverage is 11/17*100% ~= 65%. So the output should be {{"coverage": 65, "details": "ss"}}.
                   """),
            (
                "human",
                "Generated_cases are: {existing_cases},"
                "API contract is: {api_contract}"),
        ]
    )

    # model to generate cases
    case_generator = case_generator_prompt | model
    # model to check coverage
    check_coverage_executor = check_coverage_prompt | model

    # generate cases from GPT
    def generate_cases_from_gpt(self, state: State):
        res = self.case_generator.invoke({"user_input": [HumanMessage(content=state["user_input"])],
                                          "existing_cases": [HumanMessage(content=state["generated_cases"])],
                                          "api_contract": [HumanMessage(content=state["api_contract"])],
                                          "service_host": [HumanMessage(content=state["service_host"])],
                                          "expected_scenarios": [
                                              SystemMessage(content=json.dumps(self.FIELD_EXPECTED_SCENARIOS))],
                                          "case_coverage": [
                                              HumanMessage(content=json.dumps(state["case_coverage_details"]))],
                                          "custom_prompts": [HumanMessage(content=state["custom_prompts"])]
                                          })

        current_cases = state["generated_cases"]

        new_cases = convert_to_json_list_str(res.content)
        if len(current_cases) > 0 and new_cases.find("[") == 0:
            current_cases = current_cases[0:current_cases.rfind("]")] + ","
            new_cases = new_cases[1:]
        # append new cases to existing cases string
        current_cases += new_cases
        print(current_cases)
        print("--" * 20)
        # store cases to state
        return {"generated_cases": current_cases}

    # check coverage of generated cases
    def check_case_coverage(self, state: State):
        res = self.check_coverage_executor.invoke(
            {"expected_scenarios": [SystemMessage(content=json.dumps(self.FIELD_EXPECTED_SCENARIOS))],
             "existing_cases": [HumanMessage(content=state["generated_cases"])],
             "api_contract": [HumanMessage(content=state["api_contract"])]})
        print("After check coverage, output: " + res.content)
        user_input = "enough cases"

        output = convert_to_json_obj_str(res.content)
        check_result = json.loads(output)
        if int(check_result["coverage"]) <= CASE_COVERAGE_THRESHOLD:
            # if coverage is not enough, ask case generator to design missing cases
            user_input = "design more cases"

        details = check_result["details"]

        return {"user_input": user_input, "case_coverage": check_result["coverage"],
                "case_coverage_details": json.dumps(details)}

    # validate coverage if need to generate more cases or generate automation test code
    def validate(self, state: State):
        coverage_number = int(state["case_coverage"])
        print("Case coverage: " + str(coverage_number) + "%")
        print("**" * 20)
        if coverage_number >= CASE_COVERAGE_THRESHOLD:
            return "hit target"
        else:
            return "below target"

    def save_to_file(self, state: State):
        if not state["generated_cases"] or len(state["generated_cases"]) == 0:
            print("No cases to save")
            return

        case_list = json.loads(state["generated_cases"])
        file_name = case_list[0]["service_name"] + "_" + case_list[0]["api_name"] + ".csv"
        print(f"Saving cases to file [{file_name}] ...")
        test_data = []
        for case in case_list:
            row = []
            row.append(case["service_name"])
            row.append(case["service_host"])
            row.append(case["api_name"])
            row.append(case["case_name"].replace("_", " "))
            row.append(case["case_name"])
            row.append(case["http_method"])
            row.append('{"Content-Type": "application/json"}')
            row.append(case["url"])

            param_data = {}
            if "params" in case:
                param_data = {"params": case["params"]}
            elif "json" in case:
                param_data = {"json": case["json"]}
            row.append(json.dumps(param_data))
            row.append(json.dumps(case["validation"]))
            row.append("")
            row.append("")
            row.append("no")
            test_data.append(row)

        # save cases to a file for review
        save_case_to_file(file_name, test_data)
        print("Saved cases to file")

    def need_automation(self, state: State):
        if self.automate_case:
            return "yes"
        else:
            return "no"

    # generate automation test cases
    def generate_automation_cases(self, state: State):
        print("generate automation test cases")
        case_list = json.loads(state["generated_cases"])
        # generate automation test code
        for case in case_list:
            if "params" in case:
                case["params_body"] = {"params": case["params"]}
            elif "json" in case:
                case["params_body"] = {"json": case["json"]}
            else:
                case["params_body"] = {}
            case["case_description"] = case["case_name"].replace("_", " ")
            case["case_link"] = ""
            case["case_link_name"] = ""
            case["validation"] = json.dumps(case["validation"])
            case["params_body"] = json.dumps(case["params_body"])
            case["headers"] = '{"Content-Type": "application/json"}'
            add_automation_test_case(case)

    # entry point to run the workflow
    def run(self):
        # langGraph workflow
        workflow = StateGraph(State)

        workflow.set_entry_point("case_generator")

        workflow.add_node("case_generator", self.generate_cases_from_gpt)
        workflow.add_node("check_coverage", self.check_case_coverage)
        workflow.add_node("save_to_file", self.save_to_file)
        workflow.add_node("automate_case", self.generate_automation_cases)

        workflow.add_edge("case_generator", "check_coverage")
        # conditional edges to check if need to generate more cases or generate automation test code
        workflow.add_conditional_edges(
            "check_coverage", self.validate, {"below target": "case_generator", "hit target": "save_to_file"})
        workflow.add_conditional_edges(
            "save_to_file", self.need_automation, {"yes": "automate_case", "no": END})
        workflow.add_edge("automate_case", END)

        app = workflow.compile()
        result = app.invoke({"user_input": "Please design test cases", "api_contract": self.api_contract,
                             "service_host": self.service_host, "generated_cases": "", "case_coverage": 0,
                             "case_coverage_details": "", "custom_prompts": self.custom_prompts})


if __name__ == "__main__":
    api_def = """
              rpc ListOTADeviceReports(ListOTADeviceReportsRequest) returns (ListOTADeviceReportsResponse) {
    option (google.api.http) = {
      get: "/v1/ota/products/{product_name}/{model}/reports"
    };
  }
            message ListOTADeviceReportsRequest {
  string product_name = 1 [(validate.rules).string = {pattern: "^[a-zA-Z0-9]([a-z0-9A-Z\\s_-]*[a-zA-Z0-9_-])*$", min_len: 1, max_len:64}, (grpc.gateway.protoc_gen_swagger.options.openapiv2_field) = {
    min_length: 1
    max_length: 64
    pattern: "^[a-zA-Z0-9]([a-z0-9A-Z\\s_-]*[a-zA-Z0-9_-])*$"
    required: ["product_name"]
    description: "product_name is not existed in table ota_products, return 400"
  }];
  string model = 2 [(validate.rules).string = {pattern: "^[a-zA-Z0-9]([a-z0-9A-Z\\s_-]*[a-zA-Z0-9_-])*$", min_len: 1, max_len:64}, (grpc.gateway.protoc_gen_swagger.options.openapiv2_field) = {
    min_length: 1
    max_length: 64
    pattern: "^[a-zA-Z0-9]([a-z0-9A-Z\\s_-]*[a-zA-Z0-9_-])*$"
    required: ["model"]
    description: "model is not existed in table ota_product_models, return 400"
  }];
  int64 page = 3;
  int64 page_size = 4;
  repeated string filters = 5 [(validate.rules).repeated.items.string.pattern = "^(device_id|source_version_name|target_version_name|ota_state|created_at).*$"];
  string sort_by = 6 [(validate.rules).string.pattern = "^(created_at).*$"];
}"""
    api_def = api_def.replace("{", "{{").replace("}", "}}")
    # service host
    service_url = "https://myservice.com"

    business_knowledge = {
        "business": """For generated cases, service_name should my_service. AP"""
    }

    generator = LLMWorker(api_def, service_url, **business_knowledge)
    generator.run()
