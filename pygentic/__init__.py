from __future__ import annotations
import json
import sys
import os
from .chat_render import ChatRendererToString, default_template
from .llm_backends import BaseLLM, LlamaCpp, GenerationSpec
from .tools import *
from .completion import *
from .completion import RunOutOfContextError, ParentOutOfContextError
from .tool_calling import *
from .misc import Message, TextSection, ToolCallSection, ResultSection
from .loaders import FileTreeLoader, FileLoadingConfig
from .messenger import TokenArrivedEvent, GenerationCompleteEvent, messenger
from .messages import JinjaChatFactory, collate


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


class OutputDevice:
    def __call__(self, new_text):
        pass

    def on_token(self, token):
        pass


class FileOutputDevice(OutputDevice):
    def __init__(self, file_path):
        self.file_path = file_path
        self.cache = TextCache()

    def on_token(self, token):
        self.cache.fill(token)

        # todo: save only tokens of text modality
        self.append_file(token)

    def __call__(self, new_text):
        new_text = self.cache(new_text)
        if new_text:
            new_text += '\n'
        self.append_file(new_text)

    def append_file(self, text):
        with open(self.file_path, 'a') as f:
            f.write(text)


class TextCache:
    def __init__(self):
        self.buffer = ""

    def fill(self, text):
        self.buffer += text

    def __call__(self, text):
        """Returns a suffix of "text" starting with the first character after common prefix"""
        n = get_common_prefix_length(self.buffer, text)
        self.buffer = self.buffer[n:]
        return text[n:]


def get_common_prefix_length(s1, s2):
    count = 0
    for ch1, ch2 in zip(s1, s2):
        if ch1 != ch2: break
        count += 1
    
    return count


class Agent:
    default_done_tool = lambda *args, **kwargs: kwargs

    def __init__(self, llm, tools, done_tool=None, system_message="",
                 max_rounds=5, output_device=None, temp_output_device=None):
        self.llm = llm
        self.tools = tools
        self.done_tool = done_tool or self.default_done_tool
        self.system_message = system_message
        self.max_rounds = max_rounds
        self.output_device = output_device or OutputDevice()
        self.temp_output_device = temp_output_device or self.create_temp_terminal()

        self.sub_agents = {}
        self.parent = None

        self.loading_config = FileLoadingConfig.empty_config()

        self.history = []

        self.tool_use_helper = SimpleTagBasedToolUse.create_default()
        self.chat_factory = JinjaChatFactory('llama3', self.tool_use_helper)
        self.chat_renderer = self.chat_factory.get_chat_renderer()

    def create_temp_terminal(self):
        if not isinstance(self.output_device, FileOutputDevice):
            raise Exception("Cannot create FileOutputDevice for temporary terminal")

        dir_path, file_name = os.path.split(self.output_device.file_path)
        name, ext = os.path.splitext(file_name)
        new_file_path = os.path.join(dir_path, f'{name}_private.txt')

        return FileOutputDevice(new_file_path)

    def set_loading_config(self, config: FileLoadingConfig):
        self.loading_config = config

    def add_subagent(self, name, sub_agent):
        sub_agent.parent = self
        self.sub_agents = self.sub_agents or {}
        self.sub_agents[name] = sub_agent

    def __call__(self, inputs, files=None):
        # todo: allow system message to be mixed modality as well
        # todo: error counter to allow at most n attempts to call tool and give up

        tool_use_helper = self.tool_use_helper
        system_message = self.chat_factory.create_system_msg(self.system_message)

        prompt = self._prepare_prompt_message(inputs, files)

        self.output_device(system_message.content.render())
        self.output_device(prompt.content.render())

        self.completer = completer = TextCompleter(self.llm)

        completer.on_token = self._stream_to_device(tool_use_helper)

        self.history.append(system_message)
        self.history.append(prompt)

        blank_count = 0

        for _ in range(self.max_rounds):
            input_text = self.chat_renderer(self.history)
            response = completer(input_text)

            if self._blank_response(response):
                blank_count += 1

                print("GOT BLANK")

                if blank_count >= 3:
                    # when llm keeps generating blank strings, (on behalf of prompter) ask it to continue
                    msg = self.chat_factory.create_user_msg("Continue, please")
                    self.history.append(msg)
                    blank_count = 0
                    print("Generated 3 blanks!!!")
                    continue

            try:
                self._process_response(response)
            except SolutionComplete as result:
                arg_dict = result.args[0]
                done_tool_call = self.chat_factory.create_tool_call('done_tool', arg_dict)
                self.output_device(done_tool_call.content.render())
                return result.args[0]

        raise TooManyRoundsError('Too many rounds of generation')

    def _blank_response(self, response):
        return not response.replace("\n", "").strip()

    def _process_response(self, response):
        if not self.tool_use_helper.contains_tool_use(response):
            self._create_and_process_message(self.chat_factory.create_ai_msg, response)
            return

        offset, length, body = self.tool_use_helper.find(response)
        pre_tool_text = response[:offset]
        self._create_and_process_message(self.chat_factory.create_ai_msg, pre_tool_text)

        try:
            action, arg_dict = self._get_action(body)
        except InvalidJsonError as exc:
            error, body = exc.args
            self._create_and_process_message(self.chat_factory.create_raw_tool_call, body)
            self._create_and_process_message(self.chat_factory.create_tool_parse_error, error)
        else:
            self._create_and_process_message(self.chat_factory.create_tool_call, action, arg_dict)
            self._perform_action(action, arg_dict)

    def _create_and_process_message(self, create_fn, *args):
        msg = create_fn(*args)
        self.history.append(msg)
        self.output_device(msg.content.render())

    def _get_action(self, body):
        try:
            tool_name, arg_dict = self.tool_use_helper.parse(body)
            return tool_name, arg_dict
        except ValueError as e:
            raise InvalidJsonError(e.args[0], body)

    def _perform_action(self, action, arg_dict):
        if action == "done_tool":
            raise SolutionComplete(arg_dict)

        if action == "delegate":
            actor = Delegator(self, self.chat_factory)
            msg = actor(action, arg_dict)
        elif action == "clarify":
            assistant = AiAssistant(self.parent) if self.parent else NullAssistant()
            text = arg_dict["text"]
            response = assistant.ask_question(text)
            msg = self.chat_factory.create_user_msg(response)
        else:
            actor = ToolCaller(self, self.chat_factory)
            msg = actor(action, arg_dict)
        
        self.history.append(msg)
        self.output_device(msg.content.render())

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

    def backup_history(self):
        self.backup = self.history[:]

    def restore_history(self):
        self.history = self.backup[:]

    def _prepare_prompt_message(self, inputs, files):
        inputs = dict(inputs)
        files = files or []

        files_content = []
        for file_entry in files:
            path = file_entry['path']
            sections = FileTreeLoader(self.loading_config, self.chat_factory)(path)
            files_content.extend(sections)

        prompt_text = json.dumps(inputs)
        messages = files_content + [self.chat_factory.create_user_msg(prompt_text)]
        return collate(messages)


class NullAssistant:
    def ask_question(self, text):
        raise NotImplementedError


class AiAssistant(NullAssistant):
    def __init__(self, parent_agent):
        self.completer = parent_agent.completer
        self.history = parent_agent.history[:]
        self.output_device =  parent_agent.temp_output_device
        self.chat_factory = parent_agent.chat_factory

    def ask_question(self, text):
        msg = self.chat_factory.create_user_msg(text)
        self.history.append(msg)
        self.output_device(text)

        input_text = self.chat_factory.get_chat_renderer()(self.history)

        try:
            response = self.completer(input_text)
        except RunOutOfContextError as e:
            raise ParentOutOfContextError(*e.args)

        msg = self.chat_factory.create_ai_msg(response)
        self.history.append(msg)
        self.output_device(response)
        return response


class SolutionComplete(Exception):
    pass


class Delegator:
    def __init__(self, agent, chat_factory):
        self.agent = agent
        self.chat_factory = chat_factory

    def __call__(self, action, arg_dict):
        try:
            result = self._delegate(arg_dict)
            msg = self.chat_factory.create_tool_result(action, result)
        except ParentOutOfContextError as e:
            raise RunOutOfContextError(*e.args)
        except RunOutOfContextError as e:
            msg = self.chat_factory.create_tool_error(action, str(e))
        return msg

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

    def __call__(self, action, arg_dict):
        inquiry = arg_dict["text"]
        return self.assistant.ask_question(inquiry)


class ToolCaller:
    def __init__(self, agent, chat_factory):
        self.agent = agent
        self.chat_factory = chat_factory

    def __call__(self, action, arg_dict):
        tool_name = action

        try:
            result = self._use_tool(tool_name, arg_dict)
            msg = self.chat_factory.create_tool_result(tool_name, result)
        except Exception as e:
            msg = self.chat_factory.create_tool_error(tool_name, str(e))
        return msg

    def _use_tool(self, tool_name, arg_dict):
        if tool_name not in self.agent.tools:
            raise ToolDoesNotExistError(f'Tool "{tool_name}" not found', tool_name)
        try:
            return self.agent.tools[tool_name](**arg_dict)
        except TypeError as e:
            raise BadToolUseError(f'Calling tool "{tool_name}" resulted in error: {e.args[0]}')
        except Exception as e:
            raise ToolUseError(f'Calling tool "{tool_name}" resulted in error: {e.args[0]}')


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
