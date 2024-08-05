from .chat_render import ChatRendererToString, default_template
from .tool_calling import SimpleTagBasedToolUse
from .messenger import messenger, TokenArrivedEvent, GenerationCompleteEvent


def render_messages_to_string(messages, system_message=''):
    renderer = ChatRendererToString(default_template)
    renderer(system_message, messages)
    return ""


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
                raise ToolDoesNotExistError(f'Tool "{action_name}" not found', action_name)

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


def handle_delegate(agent, arg_dict, retries=3):
    name = arg_dict["name"]
    sub_agent_inputs = arg_dict["inputs"]
    sub_agent = agent.sub_agents[name]
    
    exc = None
    for _ in range(retries):
        try:
            return sub_agent(sub_agent_inputs)
        except RunOutOfContextError as e:
            exc = e
    
    raise exc


def handle_tool_use(agent, arg_dict):
    name = arg_dict["name"]
    if name not in agent.tools:
        raise ToolDoesNotExistError(f'Tool "{name}" not found')
    return agent.tools[name](**arg_dict)


def handle_failure(agent, arg_dict):
    raise Exception("Failed")


class BaseResponse:
    pending_action = "pending_action"
    solution = "solution"
    failure = "failure"
    regular_response = "regular_response"


class RegularResponse(BaseResponse):
    def __init__(self, text):
        self.text = text
        self.response_type = "regular_response"


class SolutionResponse(BaseResponse):
    def __init__(self, text, arg_dict):
        self.text = text
        self.response_type = "solution"
        self.arg_dict = arg_dict


class TextCompleter:
    def __init__(self, llm):
        self.llm = llm
        self.on_token = lambda token: token

    def __call__(self, input_text):
        raw_response = ""

        for token in self.llm(input_text):
            messenger.publish(TokenArrivedEvent(token))
            self.on_token(token)
            raw_response += token

        event_data = (raw_response, self.llm.response_data)
        messenger.publish(GenerationCompleteEvent(event_data))

        if hasattr(self.llm, "response_data") and self.llm.response_data.get("truncated"):
            raise RunOutOfContextError("LLM failed generating response: ran out of context")

        return raw_response


class RunOutOfContextError(Exception):
    pass


class ParentOutOfContextError(Exception):
    pass


class ToolDoesNotExistError(Exception):
    pass


class ToolUseError(Exception):
    pass


class InvalidJsonError(Exception):
    pass


class BadToolUseError(Exception):
    pass
