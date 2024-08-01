import requests
import json
from dataclasses import dataclass


class RequestMaker:
    def __init__(self, proxies=None):
        self.proxies = proxies or {}

    def get(self, *args, **kwargs):
        if self.proxies:
            kwargs["proxies"] = self.proxies
        return requests.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        if self.proxies:
            kwargs["proxies"] = self.proxies
        return requests.post(*args, **kwargs)


class BaseLLM:
    def __init__(self):
        self.logger = lambda x: x

    def __call__(self, text):
        raise NotImplementedError

    def add_logger(self, logger):
        self.logger = logger


class LlamaCpp(BaseLLM):
    def __init__(self, base_url, generation_spec, proxies=None):
        super().__init__()
        self.base_url = base_url

        self.generation_spec = generation_spec
        self.proxies = proxies or {}
        self.request_maker = RequestMaker(proxies)

        self.headers = {'Content-Type': 'application/json'}

        self.response_data = {}

    def __call__(self, prompt):
        sampling_config = self.generation_spec.sampling_config or {}

        clean_llm_settings(sampling_config)

        yield from self.stream_response(prompt, sampling_config)

    def stream_response(self, prompt, sampling_settings):
        stop_word = self.generation_spec.stop_word
        resp = self.start_streaming(prompt, sampling_settings, stop_word)
        line_gen = resp.iter_lines(chunk_size=1)

        for line in self.skip_empty(line_gen):
            entry = self.parse_line(line)

            should_stop = entry["stop"] and entry["stopping_word"]
            value = stop_word if should_stop else entry["content"]
            self.response_data = entry
            yield value

            if should_stop:
                break

    def start_streaming(self, prompt, sampling_settings, stop_word):
        url = f"{self.base_url}/completion"

        payload = {"prompt": prompt, "stream": True, "stop": [stop_word], "cache_prompt": True}
        payload.update(sampling_settings)

        return self.request_maker.post(url, data=json.dumps(payload), 
                                       headers=self.headers, stream=True)

    def skip_empty(self, line_generator):
        return (line for line in line_generator if line)

    def parse_line(self, line):
        line = line.decode('utf-8')
        stripped_line = line[6:]
        return json.loads(stripped_line)


@dataclass
class GenerationSpec:
    sampling_config: dict
    stop_word: str = None

    def to_dict(self):
        return self.__dict__


class ClearContextError(Exception):
    pass


class PrepareModelError(Exception):
    pass


def clean_llm_settings(llm_settings):
    clean_float_field(llm_settings, 'temperature')
    clean_float_field(llm_settings, 'top_k')
    clean_float_field(llm_settings, 'top_p')
    clean_float_field(llm_settings, 'min_p')
    clean_float_field(llm_settings, 'repeat_penalty')
    clean_int_field(llm_settings, 'n_predict')


def clean_float_field(llm_settings, field):
    """Make sure that the value of the field is float, if field exists"""
    clean_any_field(llm_settings, field, float)


def clean_int_field(llm_settings, field):
    """Make sure that the value of the field is int, if field exists"""
    clean_any_field(llm_settings, field, int)


def clean_any_field(llm_settings, field, target_type):
    """Make sure that the value of the field is of target_type if field exists"""
    if field in llm_settings:
        llm_settings[field] = target_type(llm_settings[field])
