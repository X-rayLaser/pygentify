from __future__ import annotations
from dataclasses import dataclass
import re
import json
from .chat_render import ChatRendererToString, default_template
from .llm_backends import BaseLLM, LlamaCpp, GenerationSpec
from .tools import *

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
    result = result or ''
    return f'<|tool_use_start|>{json.dumps(data)}<|tool_use_end|><|result_start|>{result}<|result_end|>'


def render_tool_use_error(tool_name, arg_dict, error=None):
    data = {'tool_name': tool_name, 'args': arg_dict}
    error = error or ''
    return f'<|tool_use_start|>{json.dumps(data)}<|tool_use_end|><|error_start|>{error}<|error_end|>'


def render_messages_to_string(messages, system_message=''):
    renderer = ChatRendererToString(default_template)
    renderer(system_message, messages)
    return ""


@dataclass
class LLMConfig:
    model: str
    context: int = 1024
    temperature: float = 0.1


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
            if action_name not in self.agent.tools:
                raise ToolDoesNotExistError(f'Tool "{action_name}" not found')

            try:
                return self.agent.tools[action_name](**arg_dict)
            except TypeError as e:
                raise BadToolUseError(f'Calling tool "{action_name}" resulted in error: {e.args[0]}')
            except Exception as e:
                raise ToolUseError(f'Calling tool "{action_name}" resulted in error: {e.args[0]}')

        return handler(self.agent, arg_dict)


class ToolUseFailedError(Exception):
    pass


def handle_clarify(agent, arg_dict):
    text = arg_dict["text"]
    return agent.parent.ask_question(text)


def handle_delegate(agent, arg_dict):
    name = arg_dict["name"]
    sub_agent_inputs = arg_dict["inputs"]
    return agent.sub_agents[name](sub_agent_inputs)


def handle_tool_use(agent, arg_dict):
    name = arg_dict["name"]
    if name not in agent.tools:
        raise ToolDoesNotExistError(f'Tool "{name}" not found')
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
        return render_tool_use_string(self.action, self.arg_dict, result)

    def render_error(self, error):
        return render_tool_use_error(self.action, self.arg_dict, error)


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


class ToolAugmentedTextCompleter:
    def __init__(self, agent, llm):
        self.agent = agent
        self.llm = llm
        self.on_token = lambda token: token

    def __call__(self, input_text):
        raw_response = ""

        for token in self.llm(input_text):
            self.on_token(token)
            raw_response += token

        return self._finalize(raw_response)

    def _finalize(self, raw_response):
        if not contains_tool_use(raw_response):
            return RegularResponse(raw_response)

        try:
            response = self._get_tool_use_response(raw_response)
        except InvalidJsonError as e:
            response = self._get_malformed_syntax_response(e)
        
        return response

    def _get_malformed_syntax_response(self, exc):
        error = exc.args[0]
        pre_tool_text = exc.args[1]
        body = exc.args[2]

        # use a specialized function for rendering malformed tool invocation
        bad_tool_use = f'<|tool_use_start|>{body}<|tool_use_end|>'
        error = f'<|tool_use_error_start|>{error}<|tool_use_error_end|>'

        response = pre_tool_text + bad_tool_use + ' ' + error
        return RegularResponse(response)

    def _get_tool_use_response(self, raw_response):
        pre_tool_text, action, arg_dict = self._get_tool_use(raw_response)
    
        if action == "done_tool":
            return SolutionResponse(pre_tool_text, arg_dict)

        response = PendingActionResponse(pre_tool_text, action, arg_dict)
        try:
            result = self._perform_action(action, arg_dict)
            resp_str = response.render_success(result)
        except Exception as e:
            resp_str = response.render_error(str(e))

        return RegularResponse(pre_tool_text + resp_str)

    def _perform_action(self, action_name, arg_dict):
        handlers = {
            'clarify': handle_clarify, 
            'delegate': handle_delegate,
            'use_tool': handle_tool_use
        }
        dispatcher = ActionDispatcher(self.agent, handlers)
        return dispatcher(action_name, arg_dict)

    def _get_tool_use(self, response):
        offset, length, body = find_tool_use(response)
        
        pre_tool_text = response[:offset]

        try:
            tool_name, arg_dict = self._parse(body)
            return pre_tool_text, tool_name, arg_dict
        except ValueError as e:
            raise InvalidJsonError(e.args[0], pre_tool_text, body)

    def _parse(self, body):
        try:
            tool_name, arg_dict = parse_tool_use(body)
            return tool_name, arg_dict
        except ValueError as e:
            body += '}'
            print("Value error, trying to recover with body:", body)
            return parse_tool_use(body)


class ToolDoesNotExistError(Exception):
    pass


class ToolUseError(Exception):
    pass


class InvalidJsonError(Exception):
    pass


class Chatbot:
    def __init__(self, system_message, prompt, completer, template, output_device):
        self.thread = Thread(system_message)
        self.thread.add_message(prompt)
        self.scratch = ""
        self.completer = completer
        self.template = template

        self.output_device = output_device
        def trigger_update(token):
            self.output_device.on_token(token)
        self.completer.on_token = trigger_update

        self.output_device(system_message)
        self.output_device(prompt)

    def respond(self, input_text):
        self.flush_scratch()
        self.thread.add_message(input_text)
        self.output_device(input_text)
        return self.generate_more()

    def generate_more(self):
        input_text = self.full_text()
        response = self.completer(input_text)
        self.scratch += response.text
        self.output_device(response.text)
        return response

    def full_text(self):
        return self.thread.render(self.template) + self.scratch

    def flush_scratch(self):
        if self.scratch:
            self.thread.add_message(self.scratch)
            self.scratch = ""


class FileOutputDevice:
    def __init__(self, file_path):
        self.file_path = file_path
        self.buffer = ""
        self.conversation_text = ""

    def on_token(self, token):
        self.buffer += token
        self.append_file(token)

    def __call__(self, new_text):
        if self.buffer and new_text.startswith(self.buffer):
            new_text = new_text[len(self.buffer):] + '\n'
        else:
            new_text += '\n'

        self.append_file(new_text)
        self.buffer = ""

    def append_file(self, text):
        with open(self.file_path, 'a') as f:
            f.write(text)


@dataclass
class Agent:
    llm: BaseLLM
    tools: dict
    done_tool: callable = lambda: ""
    system_message: str = ""
    max_rounds: int = 5
    sub_agents: dict = None
    parent: Agent = None
    output_device: FileOutputDevice = None

    def add_subagent(self, name, sub_agent):
        sub_agent.parent = self
        self.sub_agents = self.sub_agents or {}
        self.sub_agents[name] = sub_agent

    def __call__(self, inputs):
        inputs = dict(inputs)
        prompt = json.dumps(inputs)

        completer = ToolAugmentedTextCompleter(self, self.llm)
        self.completer = completer

        chatbot = Chatbot(self.system_message, prompt, completer, default_template, self.output_device)
        self.chatbot = chatbot

        for _ in range(self.max_rounds):
            response = chatbot.generate_more()

            # todo: allow number of attempts to do syntactically correct tool call
            # todo: catch error with malformed/incorrect tool usage and include the error text to messages

            if response.response_type == "solution":
                return self.done_tool(**response.arg_dict)

        raise TooManyRoundsError('Too many rounds of generation')

    def ask_question(self, text):
        response = self.chatbot.respond(text)
        if response.response_type == "failure":
            raise Exception("Giving up")

        return response.text


class TooManyRoundsError(Exception):
    pass


class BadToolUseError(Exception):
    pass
