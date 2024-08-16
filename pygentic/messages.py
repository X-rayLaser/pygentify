from dataclasses import dataclass
from typing import Union, List, Any
from .jinja_env import env
from .tool_calling import SimpleTagBasedToolUse


class Modality:
    def render(self):
        raise NotImplemented

    @property
    def text(self):
        return self.render()


@dataclass
class TextModality(Modality):
    data: str

    def render(self):
        return self.data


@dataclass
class ImageModality(Modality):
    data: bytes
    mime_type: str
    width: int
    height: int


@dataclass
class ToolCall(Modality):
    name: str
    arg_dict: dict
    renderer: callable

    def render(self):
        return self.renderer(self.name, self.arg_dict)


@dataclass
class RawToolCall(Modality):
    syntax: str
    renderer: callable

    def render(self):
        return self.renderer(self.syntax)


@dataclass
class ToolResult(Modality):
    name: str
    result: Any
    renderer: callable

    def render(self):
        return self.renderer(self.name, self.result)


@dataclass
class ToolError(Modality):
    name: str
    error: str
    renderer: callable

    def render(self):
        return self.renderer(self.name, self.error)


@dataclass
class ToolParseError(Modality):
    error: str
    renderer: callable

    def render(self):
        return self.renderer(self.error)


@dataclass
class CompositeModality(Modality):
    layout: str
    items: List[Modality]

    def render(self):
        if self.layout == "sequential":
            separator = ""
        else:
            separator = ""
        return separator.join(it.render() for it in self.items)


@dataclass
class Message:
    role: str
    content: Union[TextModality, ImageModality, ToolCall, ToolResult, CompositeModality]


# different ways of creating a family of objects rendered using different strategies
class ChatFactory:
    def create_system_msg(self, text):
        raise NotImplementedError

    def create_user_msg(self, text):
        raise NotImplementedError

    def create_ai_msg(self, text):
        raise NotImplementedError

    def create_tool_call(self, tool_name, arg_dict):
        raise NotImplementedError

    def create_tool_result(self, tool_name, result):
        raise NotImplementedError

    def create_tool_error(self, tool_name, error):
        raise NotImplementedError

    def get_chat_renderer(self):
        raise NotImplemented


class JinjaChatFactory(ChatFactory):
    """Creating messages that will be rendered using jinja2 templates
    """
    def __init__(self, arch, tool_use):
        self.arch = arch
        self.tool_use = tool_use
        
        if not tool_use:
            if arch == 'llama3':
                self.tool_use = SimpleTagBasedToolUse.create_default()
            else:
                raise Exception("Cannot find tool use backend")

    def create_system_msg(self, text):
        return self._create_text_message("system", text)

    def create_user_msg(self, text):
        return self._create_text_message("user", text)

    def create_ai_msg(self, text):
        return self._create_text_message("assistant", text)

    def _create_text_message(self, role, text):
        return Message(role=role, content=TextModality(text))

    def create_tool_call(self, tool_name, arg_dict):
        renderer = self.tool_use.render_tool_call
        modality = ToolCall(tool_name, arg_dict, renderer)
        return Message(role="tool", content=modality)

    def create_raw_tool_call(self, call_syntax):
        renderer = self.tool_use.render_raw_tool_call
        modality = RawToolCall(call_syntax, renderer)
        return Message(role="tool", content=modality)

    def create_tool_result(self, tool_name, result):
        renderer = self.tool_use.render_result
        modality = ToolResult(tool_name, result, renderer)
        return Message(role="tool", content=modality)

    def create_tool_error(self, tool_name, error):
        renderer = self.tool_use.render_error
        modality = ToolError(tool_name, error, renderer)
        return Message(role="tool", content=modality)

    def create_tool_parse_error(self, error):
        renderer = self.tool_use.render_syntax_error
        modality = ToolParseError(error, renderer)
        return Message(role="tool", content=modality)

    def get_chat_renderer(self):
        def render(messages, group_roles=True, collate_fn=None):
            if group_roles:
                collate_fn = collate_fn or collate
                messages = [collate_fn(group) for group in group_messages(messages)]

            if self.arch == 'llama3':
                template = env.get_template('llama_3.jinja')
                return template.render(messages=messages) + '<|start_header_id|>assistant<|end_header_id|>'
            else:
                raise Exception("Cannot find jinja template to render chat")

        return render


def group_messages(messages):
    if messages:
        first, *rest = messages
        role = first.role
        group = [first]
        for msg in rest:
            if msg.role == role:
                group.append(msg)
            else:
                yield group
                role = msg.role
                group = [msg]

        if group:
            yield group


def collate(messages):
    assert len(messages) > 0
    role = messages[0].role
    items = [msg.content for msg in messages]
    modality = CompositeModality("sequential", items)
    return Message(role, modality)
