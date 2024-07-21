from __future__ import annotations
from dataclasses import dataclass
import re
import json
from .chat_render import ChatRendererToString, default_template


def find_tool_use(s):
    pattern = r"\<\|tool_use_start\|\>([^<]*)<\|tool_use_end\|>"
    match = re.search(pattern, s)
    if match:
        return match.start(), len(match.group(0)), match.group(1)
    else:
        raise ToolUseNotFoundError("Tool use not found")


class ToolUseNotFoundError(Exception):
    pass


def contains_tool_use(s):
    try:
        find_tool_use(s)
        return True
    except ToolUseNotFoundError:
        return False


def parse_tool_use(text):
    try:
        data = json.loads(text)
        if 'tool_name' in data:
            return (data['tool_name'], data.get('args', {}))
        else:
            raise ValueError("Tool name not found in JSON string")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


def render_tool_use_string(tool_name, arg_dict, result=None):
    data = {'tool_name': tool_name, 'args': arg_dict}
    if result:
        data['result'] = result
    return json.dumps(data)


def render_tool_use_error(tool_name, arg_dict, error=None):
    data = {'tool_name': tool_name, 'args': arg_dict}
    if error:
        data['error'] = error
    return json.dumps(data)


def render_messages_to_string(messages, system_message=''):
    renderer = ChatRendererToString(default_template)
    renderer(system_message, messages)
    return ""


@dataclass
class LLMConfig:
    model: str
    context: int = 1024
    temperature: float = 0.1


class BaseLLM:
    def __call__(self, text):
        raise NotImplementedError


class LlamaCpp(BaseLLM):
    def __init__(self, origin, config):
        self.origin = origin
        self.config = config

    def __call__(self, text):
        return ""


class MockLLM(BaseLLM):
    def __init__(self, response):
        self.response = response

    def __call__(self, text):
        return self.response


class ActionDispatcher:
    def __init__(self, agent, action_handlers: dict):
        self.agent = agent
        self.action_handlers = action_handlers

    def __call__(self, action_name, arg_dict):
        if action_name == "failure":
            raise ToolUseFailedError

        handler = self.action_handlers.get(action_name)
        if not handler:
            if action_name in self.agent.tools:
                try:
                    return self.agent.tools[action_name](**arg_dict)
                except TypeError as e:
                    raise BadToolUseError(*e.args)
                except Exception:
                    raise
            raise BadToolUseError(f"Action handler for '{action_name}' not found")

        return handler(self.agent, arg_dict)


class ToolUseFailedError(Exception):
    pass


def handle_clarify(agent, arg_dict):
    text = arg_dict["text"]
    return agent.parent.ask_question(text)


def handle_delegate(agent, arg_dict):
    name = arg_dict["name"]
    sub_agent_inputs = arg_dict["inputs"]
    return agent.sub_agents[name](**sub_agent_inputs)


def handle_tool_use(agent, arg_dict):
    name = arg_dict["name"]
    return agent.tools[name](**arg_dict)


def handle_failure(agent, arg_dict):
    raise Exception("Failed")


class GeneratorWithRetries:
    def __init__(self, llm, max_retries=3, max_continue=3):
        self.llm = llm
        self.max_retries = max_retries
        self.max_continue = max_continue

    def __call__(self, input_text):
        response = self._try_generate(input_text, self.max_retries)
        for _ in range(self.max_continue):
            if self.incomplete_response(response):
                response += self._try_generate(response, self.max_retries)
        return response

    def incomplete_response(self, text):
        # todo: implement this
        return False

    def _try_generate(self, input_text, tries):
        try:
            return self.llm(input_text)
        except Exception:
            if tries > 0:
                return self._try_generate(input_text, tries - 1)
            else:
                raise


class Thread:
    def __init__(self, system_message):
        self.system_message = system_message
        self.messages = []

    def add_message(self, text):
        self.messages.append(text)

    def append_previous(self, text):
        self.messages[-1] += text

    def render(self, template):
        renderer = ChatRendererToString(template)
        return renderer(self.system_message, self.messages)


class BaseResponse:
    pending_action = "pending_action"
    solution = "solution"
    failure = "failure"
    regular_response = "regular_response"


class RegularResponse(BaseResponse):
    def __init__(self, text):
        self.text = text
        self.response_type = "regular_response"


class PendingActionResponse(BaseResponse):
    def __init__(self, pre_tool_text, action, arg_dict):
        self.pre_tool_text = pre_tool_text
        self.response_type = "pending_action"
        self.action = action
        self.arg_dict = arg_dict

    def render_success(self, result):
        tool_use_str = render_tool_use_string(self.action, self.arg_dict, result)
        return self.pre_tool_text + tool_use_str

    def render_error(self, error):
        tool_use_str = render_tool_use_error(self.action, self.arg_dict, error)
        return self.pre_tool_text + tool_use_str


class SolutionResponse(BaseResponse):
    def __init__(self, text, arg_dict):
        self.text = text
        self.response_type = "solution"
        self.arg_dict = arg_dict


class FailureResponse(BaseException):
    def __init__(self, text, error_dict):
        self.text = text
        self.response_type = "failure"
        self.error_dict = error_dict


class LLMChat:
    def __init__(self, llm, system_message, prompt, template) -> None:
        self.thread = Thread(system_message)
        self.thread.add_message(prompt)
        self.llm = llm
        self.template = template

    def ask_question(self, text):
        self.thread.add_message(text)
        response = self._generate_completion(self.thread)
        self.thread.add_message(response)
        return response

    def continue_thought(self):
        response = self._generate_completion(self.thread)
        if response.response_type != "pending_action":
            self.thread.append_previous(response.text)
        return response

    def add_action_result(self, response, result):
        resp_str = response.render_success(result)
        self.thread.append_previous(resp_str)

    def add_action_error(self, response, error):
        resp_str = response.render_error(error)
        self.thread.append_previous(resp_str)

    def _generate_completion(self, thread):
        thread_text = thread.render(template=self.template)
        response = self.llm(thread_text)

        try:
            pre_tool_text, tool_name, arg_dict = self._get_tool_use(response)
            if tool_name == "done_tool":
                response = SolutionResponse(pre_tool_text, arg_dict)
            else:
                response = PendingActionResponse(pre_tool_text, tool_name, arg_dict)
        except ToolUseNotFoundError:
            response = RegularResponse(response)
        return response

    def _get_tool_use(self, response):
        try:
            offset, length, body = find_tool_use(response)
            tool_name, arg_dict = parse_tool_use(body)
            pre_tool_text = response[:offset]
            return pre_tool_text, tool_name, arg_dict
        except ValueError:
            raise BadToolUseError


@dataclass
class Agent:
    llm: BaseLLM
    tools: dict
    done_tool: callable = lambda: ""
    system_message: str = ""
    max_rounds: int = 5
    sub_agents: dict = None
    parent: Agent = None

    def add_subagent(self, name, sub_agent):
        sub_agent.parent = self
        self.sub_agents = self.sub_agents or {}
        self.sub_agents[name] = sub_agent

    def __call__(self, inputs):
        inputs = dict(inputs)
        prompt = json.dumps(inputs)

        template = default_template
        chat = LLMChat(self.llm, self.system_message, prompt, template)
        self.chat = chat

        for _ in range(self.max_rounds):
            response = chat.continue_thought()

            # todo: allow number of attempts to do syntactically correct tool call
            # todo: catch error with malformed/incorrect tool usage and include the error text to messages

            if response.response_type == "solution":
                return self.done_tool(**response.arg_dict)

            if response.response_type == "failure":
                raise Exception("Giving up")

            if response.response_type == "pending_action":
                try:
                    result = self._perform_action(response.action, response.arg_dict)
                    chat.add_action_result(response, result)
                except Exception as e:
                    chat.add_action_error(response, str(e))

        raise TooManyRoundsError('Too many rounds of generation')

    def _perform_action(self, action_name, arg_dict):
        handlers = {
            'clarify': handle_clarify, 
            'delegate': handle_delegate,
            'use_tool': handle_tool_use
        }
        dispatcher = ActionDispatcher(self, handlers)
        return dispatcher(action_name, arg_dict)

    def ask_question(self, text):
        return self.chat.ask_question(text)


class TooManyRoundsError(Exception):
    pass


class BadToolUseError(Exception):
    pass
