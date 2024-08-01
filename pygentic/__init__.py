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


class Chatbot:
    def __init__(self, system_message: Message, prompt: Message, completer, template, output_device):
        self.thread = Thread(system_message)
        self.thread.add_message(prompt)
        self.scratch = []
        self.completer = completer
        self.template = template

        self.output_device = output_device
        def trigger_update(token):
            self.output_device.on_token(token)
        self.completer.on_token = trigger_update

        self.output_device(str(system_message))
        self.output_device(str(prompt))

    def respond(self, input_text: str):
        self.flush_scratch()

        msg = Message.text_message(input_text)
        self.thread.add_message(msg)
        self.output_device(input_text)
        return self.generate_more()

    def generate_more(self):
        input_text = self.full_text()
        response = self.completer(input_text)

        # todo: in general, this may not be just text and may need to be parsed into different sections
        # each having their own modality (e.g. interleaving text with images)
        section = TextSection(response.text)
        self.scratch.append(section)
        self.output_device(response.text)
        return response

    def full_text(self):
        return self.thread.render(self.template) + str(Message(self.scratch))

    def flush_scratch(self):
        if self.scratch:
            self.thread.add_message(Message(self.scratch))
            self.scratch = []


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
        system_message = Message.text_message(self.system_message)
        prompt = self._prepare_prompt_message(inputs, files)

        tool_use_helper = SimpleTagBasedToolUse.create_default()
        completer = ToolAugmentedTextCompleter(self, self.llm, tool_use_helper)
        self.completer = completer

        chatbot = Chatbot(system_message, prompt, completer, default_template, self.output_device)
        self.chatbot = chatbot

        for _ in range(self.max_rounds):
            response = chatbot.generate_more()

            # todo: allow number of attempts to do syntactically correct tool call
            # todo: catch error with malformed/incorrect tool usage and include the error text to messages

            if response.response_type == "solution":
                return self.done_tool(**response.arg_dict)

        raise TooManyRoundsError('Too many rounds of generation')

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

    def ask_question(self, text):
        try:
            response = self.chatbot.respond(text)
        except RunOutOfContextError as e:
            raise ParentOutOfContextError(*e.args)

        if response.response_type == "failure":
            raise Exception("Giving up")

        return response.text


class TokenBudget:
    def __init__(self, quota):
        self.quota = quota
        self.n_tokens = 0

    def increment(self, n=1):
        self.n_tokens += n
        self._check()

    def _check(self):
        total = self.n_tokens + self.n_staged
        if total > self.quota:
            print(f'Total amount of generated tokens ({total}) has exceeded the quota ({self.quota})')
            sys.exit()


class AgentRunner:
    def __init__(self, agent, max_eval, max_gen, max_total):
        self.agent = agent
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
                print(f'Total amount of generated tokens ({total}) has exceeded the quota ({max_total})')
                sys.exit()

        messenger.subscribe(TokenArrivedEvent.etype, on_token)
        messenger.subscribe(GenerationCompleteEvent.etype, on_complete)

    def __call__(self, inputs, files=None):
        return self.agent(inputs, files)


class TooManyRoundsError(Exception):
    pass


class BudgetExceededError(Exception):
    pass
