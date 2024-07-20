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
                return self.agent.tools[action_name](**arg_dict)
            raise BadToolUseError(f"Action handler for '{action_name}' not found")

        return handler(self.agent, arg_dict)


class ToolUseFailedError(Exception):
    pass


def handle_clarify(agent, arg_dict):
    text = arg_dict["text"]
    return agent.parent.send_message(text)


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
    def __init__(self, llm, system_message, max_retries=3, max_continue=3):
        self.llm = llm
        self.system_message = system_message
        self.max_retries = max_retries
        self.max_continue = max_continue

    def __call__(self, messages):
        text = render_messages_to_string(messages, self.system_message)

        response = self._try_generate(text, self.max_retries)
        for _ in range(self.max_continue):
            if self.incomplete_response(response):
                response += self._try_generate(response, self.max_retries)
        return response

    def incomplete_response(self, text):
        # todo: implement this
        return False

    def _try_generate(self, text, tries):
        try:
            return self.llm(text)
        except Exception:
            if tries > 0:
                return self._try_generate(text, tries - 1)
            else:
                raise


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

        self.messages = messages = [prompt]

        handlers = {
            'clarify': handle_clarify, 
            'delegate': handle_delegate,
            'use_tool': handle_tool_use
        }
        dispatcher = ActionDispatcher(self, handlers)

        generator = GeneratorWithRetries(self.llm, self.system_message)

        for _ in range(self.max_rounds):
            response = generator(messages)

            # todo: allow regular response without tool use syntax
            # todo: allow number of attempts to do syntactically correct tool call
            # todo: catch error with malformed/incorrect tool usage and include the error text to messages

            pre_tool_text, tool_name, arg_dict = self._get_tool_use(response)

            if tool_name == "done_tool":
                return self.done_tool(**arg_dict)

            result = dispatcher(action_name=tool_name, arg_dict=arg_dict)
            
            tool_use_with_result = render_tool_use_string(tool_name, arg_dict, result)
            messages.append(pre_tool_text + tool_use_with_result)

        raise TooManyRoundsError('Too many rounds of generation')

    def _get_tool_use(self, response):
        try:
            offset, length, body = find_tool_use(response)
            tool_name, arg_dict = parse_tool_use(body)
            pre_tool_text = response[:offset]
            return pre_tool_text, tool_name, arg_dict
        except ToolUseNotFoundError as e:
            raise BadToolUseError(f"Invalid tool use syntax: {e}")
        except ValueError:
            raise BadToolUseError

    def send_message(self, text):
        generator = GeneratorWithRetries(self.llm, self.system_message)
        self.messages.append(text)
        response = generator(self.messages)
        self.messages.append(response)
        return response


class TooManyRoundsError(Exception):
    pass


class BadToolUseError(Exception):
    pass
