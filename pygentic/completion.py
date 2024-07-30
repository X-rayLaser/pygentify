from .chat_render import ChatRendererToString, default_template
from .tool_calling import SimpleTagBasedToolUse


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


class ToolAugmentedTextCompleter:
    def __init__(self, agent, llm, tool_use_helper=None):
        self.agent = agent
        self.llm = llm
        self.tool_use_helper = tool_use_helper or SimpleTagBasedToolUse.create_default()
        self.on_token = lambda token: token

    def __call__(self, input_text):
        raw_response = ""

        for token in self.llm(input_text):
            self.on_token(token)
            raw_response += token

        return self._finalize(raw_response)

    def _finalize(self, raw_response):
        if not self.tool_use_helper.contains_tool_use(raw_response):
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

        resp_str = self.tool_use_helper.render_with_syntax_error(body, error)
        response = pre_tool_text + resp_str
        return RegularResponse(response)

    def _get_tool_use_response(self, raw_response):
        pre_tool_text, action, arg_dict = self._get_tool_use(raw_response)
    
        if action == "done_tool":
            return SolutionResponse(pre_tool_text, arg_dict)

        try:
            result = self._perform_action(action, arg_dict)
            resp_str = self.tool_use_helper.render_with_success(action, arg_dict, result)
        except Exception as e:
            resp_str = self.tool_use_helper.render_with_error(action, arg_dict, str(e))

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
        offset, length, body = self.tool_use_helper.find(response)
        
        pre_tool_text = response[:offset]

        try:
            tool_name, arg_dict = self._parse(body)
            return pre_tool_text, tool_name, arg_dict
        except ValueError as e:
            raise InvalidJsonError(e.args[0], pre_tool_text, body)

    def _parse(self, body):
        try:
            tool_name, arg_dict = self.tool_use_helper.parse(body)
            return tool_name, arg_dict
        except ValueError as e:
            body += '}'
            print("Value error, trying to recover with body:", body)
            # todo: this is bad, if previous error happened because of wrong arguments, adding } will definitely not help
            # and will likely result in a cryptic json decode error, therefore we should instead try catch again and 
            # reraise the original error
            try:
                return self.tool_use_helper.parse(body)
            except:
                raise e


class ToolDoesNotExistError(Exception):
    pass


class ToolUseError(Exception):
    pass


class InvalidJsonError(Exception):
    pass


class BadToolUseError(Exception):
    pass
