import re
import json
from dataclasses import dataclass
from inspect import signature
from .chat_render import ChatRendererToString, default_template
from .jinja_env import env


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


class ToolUse:
    def find(self, s):
        raise NotImplementedError

    def contains_tool_use(self, s):
        try:
            self.find(s)
            return True
        except ToolUseNotFoundError:
            return False

    def parse(self, text):
        raise NotImplementedError

    def render_with_success(self, tool_name, arg_dict, result=None):
        raise NotImplementedError

    def render_with_error(self, tool_name, arg_dict, error=None):
        raise NotImplementedError


@dataclass
class GenericToolUse(ToolUse):
    test: str
    call_template: str
    success_template: str
    error_template: str
    syntax_error_template: str = ""

    def find(self, s):
        pattern = self.test
        match = re.search(pattern, s)
        if match:
            return match.start(), len(match.group(0)), match.group(1)
        else:
            raise ToolUseNotFoundError("Tool use not found")

    def parse(self, text):
        try:
            data = json.loads(text)
            if 'tool_name' in data:
                return (data['tool_name'], data.get('args', {}))
            else:
                raise ValueError("Tool name not found in JSON string")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

    def render_with_success(self, tool_name, arg_dict, result=None):
        data = {'tool_name': tool_name, 'args': arg_dict}
        result = result or ''
        body = json.dumps(data)

        return self.success_template.format(body, result)

    def render_with_error(self, tool_name, arg_dict, error=None):
        data = {'tool_name': tool_name, 'args': arg_dict}
        error = error or ''
        body = json.dumps(data)

        return self.error_template.format(body, error)

    def render(self, tool_name, arg_dict):
        data = {'tool_name': tool_name, 'args': arg_dict}
        body = json.dumps(data)
        return self.call_template.format(body)

    def render_with_syntax_error(self, body, error):
        error = error or ''
        template = self.syntax_error_template or self.error_template
        return template.format(body, error)


class SimpleTagBasedToolUse(GenericToolUse):
    def __init__(self, start_tag, end_tag, result_start_tag, result_end_tag,
                 error_start_tag, error_end_tag):
        def escape(s):
            escape_chars = '<|>'
            for ch in escape_chars:
                s = s.replace(ch, "\\" + ch)
            return s

        success_template = f'{start_tag}{{}}{end_tag}{result_start_tag}{{}}{result_end_tag}'
        call_template = f'{start_tag}{{}}{end_tag}'
        error_template = f'{start_tag}{{}}{end_tag}{error_start_tag}{{}}{error_end_tag}'


        self.start_tag = start_tag
        self.end_tag = end_tag

        start_tag = escape(start_tag)
        end_tag = escape(end_tag)
        error_start_tag = escape(error_start_tag)
        error_end_tag = escape(error_end_tag)

        test = f"{start_tag}([^<]*){end_tag}"
        super().__init__(test, call_template, success_template, error_template)

    @classmethod
    def create_default(cls):
        return cls(start_tag="<|tool_use_start|>",
                   end_tag="<|tool_use_end|>",
                   result_start_tag="<|result_start|>",
                   result_end_tag="<|result_end|>",
                   error_start_tag="<|error_start|>",
                   error_end_tag="<|error_end|>")

    def parse(self, text):
        try:
            return super().parse(text)
        except ValueError as e:
            text += '}'
            print("Value error, trying to recover with body:", text)
            # todo: even more robust behaviour, auto-correct more errors
            # todo: consider to use custom recovery strategies for fixing simple cases
            try:
                return super().parse(text)
            except:
                raise e


tool_registry = {}


class ToolRegistrator:
    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        tool_registry[self.name] = func
        return func


def register(name=None):
    def decorator(func):
        func_name = name or func.__name__
        tool_registry[func_name] = func
        return func
    
    return decorator


def default_tool_use_backend():
    return SimpleTagBasedToolUse.create_default()
