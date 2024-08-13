from __future__ import annotations
import json
import sys
from .chat_render import ChatRendererToString, default_template
from .llm_backends import BaseLLM, LlamaCpp, GenerationSpec
from .tools import *
from .completion import *
from .completion import RunOutOfContextError, ParentOutOfContextError
from .tool_calling import *
from .misc import Message, TextSection
from .loaders import FileTreeLoader, FileLoadingConfig
from .messenger import TokenArrivedEvent, GenerationCompleteEvent, messenger


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

    def add_message(self, msg):
        self.messages.append(msg)

    def render(self, template):
        renderer = ChatRendererToString(template)
        return renderer(self.system_message, self.messages)


class ChatHistory:
    def __init__(self, system_message: Message, template):
        # todo: merge with thread
        self.thread = Thread(system_message)
        self.template = template

    def add_prompter_message(self, text):
        assert len(self.thread.messages) % 2 == 0
        self._add_message()

    def add_ai_message(self, text):
        assert len(self.thread.messages) % 2 == 1
        self._add_message()

    def _add_message(self, text):
        msg = Message.text_message(text)
        self.thread.add_message(msg)

    def full_text(self):
        return self.thread.render(self.template)

    def clone(self):
        history = ChatHistory(system_message="", template=self.template)
        history.thread.messages = self.thread.messages[:]
        history.thread.system_message = self.thread.system_message
        return history


class Scratchpad:
    def __init__(self):
        self.scratch = []
    
    def write_back(self, text):
        self.scratch.append(TextSection(text))

    def flush(self):
        res = self.to_message()
        self.scratch = []
        return res
    
    def to_text(self):
        return str(self.to_message())

    def to_message(self):
        return Message(self.scratch)


class OutputDevice:
    def __call__(self, new_text):
        pass

    def on_token(self, token):
        pass


class FileOutputDevice(OutputDevice):
    def __init__(self, file_path):
        self.file_path = file_path
        self.buffer = ""
        self.conversation_text = ""

    def on_token(self, token):
        self.buffer += token

        # todo: save only tokens of text modality
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


class Agent:
    default_done_tool = lambda *args, **kwargs: kwargs

    def __init__(self, llm, tools, done_tool=None, system_message="",
                 max_rounds=5, output_device=None):
        self.llm = llm
        self.tools = tools
        self.done_tool = done_tool or self.default_done_tool
        self.system_message = system_message
        self.max_rounds = max_rounds
        self.output_device = output_device or OutputDevice()
        self.sub_agents = {}
        self.parent = None

        self.loading_config = FileLoadingConfig.empty_config()

        self.history = ChatHistory(system_message, default_template)
        self.scratch = Scratchpad()

    def set_loading_config(self, config: FileLoadingConfig):
        self.loading_config = config

    def add_subagent(self, name, sub_agent):
        sub_agent.parent = self
        self.sub_agents = self.sub_agents or {}
        self.sub_agents[name] = sub_agent

    def __call__(self, inputs, files=None):
        # todo: allow system message to be mixed modality as well
        # todo: error counter to allow at most n attempts to call tool and give up
        system_message = Message.text_message(self.system_message)
        prompt = self._prepare_prompt_message(inputs, files)

        self.output_device(str(system_message))
        self.output_device(str(prompt))

        tool_use_helper = SimpleTagBasedToolUse.create_default()
        self.completer = completer = TextCompleter(self.llm)

        completer.on_token = self._stream_to_device(tool_use_helper)

        self.history.add_prompter_message(prompt)

        processor = self._prepare_processor(tool_use_helper)

        for _ in range(self.max_rounds):
            input_text = self.history.full_text() + self.scratch.to_text()
            response = completer(input_text)

            if not self.tool_use_helper.contains_tool_use(response):
                self.scratch.write_back(response)
                self.output_device(response)
                continue

            try:
                response = processor(response)
            except SolutionComplete as result:
                arg_dict = result.args[0]
                done_tool_call = tool_use_helper.render('done_tool', arg_dict)
                self.output_device(done_tool_call)
                return result.args[0]

        raise TooManyRoundsError('Too many rounds of generation')

    def _stream_to_device(self, tool_use_helper):
        # todo: refactor
        tool_use_seq = False
        buffer = ""
        def on_token(token):
            nonlocal tool_use_seq, buffer
            buffer += token

            #print(tool_use_helper.start_tag, buffer[-40:], bool(tool_use_helper.start_tag in buffer))

            if not tool_use_seq:
                self.output_device.on_token(token)

            if tool_use_helper.start_tag in buffer:
                tool_use_seq = True
                idx = buffer.index(tool_use_helper.start_tag)
                buffer = buffer[idx + len(buffer):]
            
            if tool_use_helper.end_tag in buffer:
                tool_use_seq = False
                idx = buffer.index(tool_use_helper.end_tag)
                buffer = buffer[idx + len(buffer):]
        return on_token

    def _prepare_processor(self, tool_use_helper):
        def error_handler(error, pre_tool_text, body, resp_str):
            response = pre_tool_text + resp_str
            self.scratch.write_back(response)
            self.output_device(response)

        def do_before_delegate(pre_tool_text, action, arg_dict):
            tool_call_syntax = self.tool_use_helper.render(action, arg_dict)
            self.scratch.write_back(pre_tool_text + tool_call_syntax)

        def do_after_delegate(action_response):
            response = action_response.pre_tool_text + action_response.response
            self.scratch.scratch.pop()
            self.scratch.write_back(response)
            self.output_device(response)

        def do_before_clarify(pre_tool_text, action, arg_dict):
            inquiry = arg_dict["text"]

            self.scratch.write_back(pre_tool_text)
            self.scratch.write_back(inquiry)
            msg = self.scratch.flush()
            self.history.add_ai_message(str(msg))

        def do_after_clarify(self, action_response):
            response = action_response.pre_tool_text + action_response.response
            self.history.add_prompter_message(action_response.response)
            self.output_device(response)

        def do_after_tool(self, action_response):
            self.scratch.write_back(action_response.pre_tool_text + action_response.response)
            self.output_device(action_response.pre_tool_text + action_response.response)

        # todo: write assistant
        assistant = None

        processor = ResponseProcessor(self, tool_use_helper, assistant, error_handler)

        processor.subscribe_pre("delegate", do_before_delegate)
        processor.subscribe_pre("clarify", do_before_clarify)
        processor.subscribe_post("delegate", do_after_delegate)
        processor.subscribe_post("clarify", do_after_clarify)
        processor.subscribe_post("tool_use", do_after_tool)
        return processor

    def ask_question(self, text):
        try:
            self.output_device(text)
            self.history.add_message(text)

            input_text = self.history.full_text()
            response = self.completer(input_text)
            self.history.write(response)
            self.output_device(response)
        except RunOutOfContextError as e:
            raise ParentOutOfContextError(*e.args)

        return response

    def backup_history(self):
        self.backup = self.history.clone()

    def restore_history(self):
        self.history = self.backup.clone()

    def _prepare_prompt_message(self, inputs, files):
        inputs = dict(inputs)
        files = files or []

        files_content = []
        for file_entry in files:
            path = file_entry['path']
            sections = FileTreeLoader(self.loading_config)(path)
            files_content.extend(sections)

        prompt_text = json.dumps(inputs)
        prompt_sections = files_content + [TextSection(prompt_text)]
        return Message(prompt_sections)


class Assistant:
    def __init__(self, completer, history, scratch):
        self.completer = completer
        self.history = history.clone()
        self.scratch = scratch.clone()

    def ask_question(self, text):
        msg = self.scratch.flush()
        # todo: we need to check that scratch is empty
        self.history.add_ai_message(str(msg))
        self.history.add_prompter_message(text)
        input_text = self.history.full_text() + self.scratch.to_text()
        response = self.completer(input_text)
        self.history.add_ai_message(response)
        return response


class ResponseProcessor:
    def __init__(self, agent, tool_helper, assistant, raw_handler=None, error_handler=None):
        self.agent = agent
        self.tool_use_helper = tool_helper
        self.assistant = assistant
        self.pre_handlers = {}
        self.post_handlers = {}

        self.raw_handler = raw_handler
        self.error_handler = error_handler

    def subscribe_pre(self, event, handler):
        self.pre_handlers[event] = handler

    def subscribe_post(self, event, handler):
        self.post_handlers[event] = handler

    def __call__(self, response):
        if not self.tool_use_helper.contains_tool_use(response):
            if self.raw_handler:
                self.raw_handler(response)
            return response

        try:
            pre_tool_text, action, arg_dict = self._get_action(response)
        except InvalidJsonError as exc:
            error, pre_tool_text, body = exc.args
            resp_str = self.tool_use_helper.render_with_syntax_error(body, error)
            response = pre_tool_text + resp_str

            if self.error_handler:
                self.error_handler(error, pre_tool_text, body, resp_str)
        else:
            response = self._perform_action(pre_tool_text, action, arg_dict)
        return response

    def _perform_action(self, pre_tool_text, action, arg_dict):
        if action == "done_tool":
            raise SolutionComplete(arg_dict)

        actors = {
            "delegate": Delegator(self.agent, self.tool_use_helper),
            "clarify": Clarifier(self.assistant)
        }

        actor = actors.get(action, ToolCaller(self.agent, self.tool_use_helper))

        tool_use_action = "tool_use"
        pre_handler = self.pre_handlers.get(action, self.pre_handlers.get(tool_use_action))
        post_handler = self.post_handlers.get(action, self.post_handlers.get(tool_use_action))

        if pre_handler:
            pre_handler(pre_tool_text, action, arg_dict)

        response = actor(pre_tool_text, action, arg_dict)

        if post_handler:
            post_handler(ActionResponse(response, pre_tool_text, action, arg_dict))
        return response

    def _get_action(self, response):
        offset, length, body = self.tool_use_helper.find(response)
        
        pre_tool_text = response[:offset]

        try:
            tool_name, arg_dict = self.tool_use_helper.parse(body)
            return pre_tool_text, tool_name, arg_dict
        except ValueError as e:
            raise InvalidJsonError(e.args[0], pre_tool_text, body)


class SolutionComplete(Exception):
    pass


class ClarificationResponse:
    def __init__(self, response):
        self.response = response


class Delegator:
    def __init__(self, agent, tool_use_helper):
        self.agent = agent
        self.tool_use_helper = tool_use_helper

    def __call__(self, pre_tool_text, action, arg_dict):
        try:
            result = self._delegate(arg_dict)
            response = self.tool_use_helper.render_with_success(action, arg_dict, result)
        except ParentOutOfContextError as e:
            raise RunOutOfContextError(*e.args)
        except RunOutOfContextError as e:
            response = self.tool_use_helper.render_with_error(action, arg_dict, str(e))
        return response

    def _delegate(self, arg_dict, retries=3):
        name = arg_dict["name"]
        sub_agent_inputs = arg_dict["inputs"]
        sub_agent = self.agent.sub_agents[name]
        self.agent.backup_history()
        exc = None
        for _ in range(retries):
            self.agent.restore_history()

            try:
                return sub_agent(sub_agent_inputs)
            except RunOutOfContextError as e:
                # todo: needs a way to clear conversation between parent and child before retry
                exc = e

        raise exc


class Clarifier:
    def __init__(self, assistant):
        self.assistant = assistant

    def __call__(self, pre_tool_text, action, arg_dict):
        inquiry = arg_dict["text"]
        return self.assistant.ask_question(inquiry)


class ToolCaller:
    def __init__(self, agent, tool_use_helper):
        self.agent = agent
        self.tool_use_helper = tool_use_helper

    def __call__(self, pre_tool_text, action, arg_dict):
        tool_name = action

        try:
            result = self._use_tool(tool_name, arg_dict)
            resp_str = self.tool_use_helper.render_with_success(tool_name, arg_dict, result)
        except Exception as e:
            resp_str = self.tool_use_helper.render_with_error(tool_name, arg_dict, str(e))
        return resp_str

    def _use_tool(self, tool_name, arg_dict):
        if tool_name not in self.agent.tools:
            raise ToolDoesNotExistError(f'Tool "{tool_name}" not found', tool_name)
        try:
            return self.agent.tools[tool_name](**arg_dict)
        except TypeError as e:
            raise BadToolUseError(f'Calling tool "{tool_name}" resulted in error: {e.args[0]}')
        except Exception as e:
            raise ToolUseError(f'Calling tool "{tool_name}" resulted in error: {e.args[0]}')


@dataclass
class ActionResponse:
    response: str
    pre_tool_text: str
    action: str
    arg_dict: dict


class UnknownActionError(Exception):
    pass


class TokenBudget:
    def __init__(self, quota):
        self.quota = quota
        self.n_tokens = 0

    def increment(self, n=1):
        self.n_tokens += n
        self._check()

    def _check(self):
        total = self.n_tokens
        if total > self.quota:
            print(f'Aborted the script. Total # of generated tokens ({total}) has exceeded the quota ({self.quota})')
            sys.exit()


def run_agent(agent, inputs, files=None, max_eval=100000, max_gen=10000, max_total=100000):
    agent = agent
    input_budget = TokenBudget(max_eval)
    output_budget = TokenBudget(max_gen)
    total_budget = TokenBudget(max_total)

    def on_token(token):
        output_budget.increment()

    def on_complete(data):
        _, response_data = data
        num_eval = response_data["tokens_evaluated"]

        input_budget.increment(num_eval)
        
        total = input_budget.n_tokens + output_budget.n_tokens
        if total > max_total:
            print(f'Aborted the script. Total # of generated tokens ({total}) has exceeded the quota ({max_total})')
            sys.exit()

    messenger.subscribe(TokenArrivedEvent.etype, on_token)
    messenger.subscribe(GenerationCompleteEvent.etype, on_complete)

    return agent(inputs, files)


class TooManyRoundsError(Exception):
    pass


class BudgetExceededError(Exception):
    pass
