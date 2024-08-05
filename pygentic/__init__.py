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

    def append_previous(self, section):
        self.messages[-1].append(section)

    def render(self, template):
        renderer = ChatRendererToString(template)
        return renderer(self.system_message, self.messages)


class ChatHistory:
    def __init__(self, system_message: Message, prompt: Message, template):
        self.thread = Thread(system_message)
        self.thread.add_message(prompt)
        self.scratch = []
        self.template = template

    def write(self, text):
        section = TextSection(text)
        self.scratch.append(section)

    def erase_last_section(self):
        self.scratch.pop()

    def add_message(self, text):
        self.flush_scratch()
        msg = Message.text_message(text)
        self.thread.add_message(msg)

    def full_text(self):
        return self.thread.render(self.template) + str(Message(self.scratch))

    def flush_scratch(self):
        if self.scratch:
            self.thread.add_message(Message(self.scratch))
            self.scratch = []

    def clone(self):
        history = ChatHistory(system_message="", prompt="", template=self.template)
        history.thread.messages = self.thread.messages[:]
        history.thread.system_message = self.thread.system_message
        history.scratch = self.scratch[:]
        return history


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
        completer = TextCompleter(self.llm)
        self.completer = completer

        completer.on_token = self._stream_to_device(tool_use_helper)

        history = ChatHistory(system_message, prompt, default_template)
        self.history = history

        processor = ResponseProcessor(self, tool_use_helper)

        for _ in range(self.max_rounds):
            input_text = history.full_text()
            response = completer(input_text)

            try:
                response = processor(response)
            except SolutionComplete as result:
                arg_dict = result.args[0]
                done_tool_call = tool_use_helper.render('done_tool', arg_dict)
                self.output_device(done_tool_call)
                return result.args[0]

            history.write(response)
            self.output_device(response)

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


class ResponseProcessor:
    def __init__(self, agent, tool_helper):
        self.agent = agent
        self.tool_use_helper = tool_helper

    def __call__(self, response):
        action_handler = ActionHandler(self.agent, self.tool_use_helper)
        if not self.tool_use_helper.contains_tool_use(response):
            return response

        try:
            pre_tool_text, action, arg_dict = self._get_tool_use(response)
        except InvalidJsonError as exc:
            error, pre_tool_text, body = exc.args
            resp_str = self.tool_use_helper.render_with_syntax_error(body, error)
            response = pre_tool_text + resp_str
        else:
            response = action_handler(pre_tool_text, action, arg_dict)
        return response

    def _get_tool_use(self, response):
        offset, length, body = self.tool_use_helper.find(response)
        
        pre_tool_text = response[:offset]

        try:
            tool_name, arg_dict = self.tool_use_helper.parse(body)
            return pre_tool_text, tool_name, arg_dict
        except ValueError as e:
            raise InvalidJsonError(e.args[0], pre_tool_text, body)


class SolutionComplete(Exception):
    pass


class ActionHandler:
    def __init__(self, agent, tool_use_helper):
        self.agent = agent
        self.tool_use_helper = tool_use_helper

    def __call__(self, pre_tool_text, action, arg_dict):
        if action == "done_tool":
            raise SolutionComplete(arg_dict)

        if action == "delegate":
            # todo: this is consfusing
            tool_call = self.tool_use_helper.render(action, arg_dict)
            partial_text = pre_tool_text + tool_call
            self.agent.history.write(partial_text)
            response = self._handle_delegate(action, arg_dict)
            response = pre_tool_text + response
            self.agent.history.erase_last_section()
        elif action == "clarify":
            # todo: and this as well
            text = arg_dict["text"]
            tool_call = self.tool_use_helper.render(action, arg_dict)
            response = self.agent.parent.ask_question(text)
            self.agent.history.write(pre_tool_text)
            self.agent.history.flush()
        else:
            response = pre_tool_text + self._handle_tool_use(action, arg_dict)
        
        return response

    def _handle_delegate(self, action, arg_dict):
        try:
            result = self._delegate(arg_dict)
            response = self.tool_use_helper.render_with_success(action, arg_dict, result)
        except ParentOutOfContextError as e:
            raise RunOutOfContextError(*e.args)
        except RunOutOfContextError as e:
            response = self.tool_use_helper.render_with_error(action, arg_dict, str(e))
        return response

    def _handle_tool_use(self, tool_name, arg_dict):
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
