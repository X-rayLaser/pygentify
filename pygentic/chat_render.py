from .jinja_env import env

default_template = "llama_3.jinja"


class ChatRenderer:
    def __init__(self, template_name, use_bos=False):
        self.template_name = template_name
        self.use_bos = use_bos

    def __call__(self, system_message, messages):
        raise NotImplementedError


class ChatRendererToString(ChatRenderer):
    def __call__(self, system_message, messages):
        template = env.get_template(self.template_name)
        return template.render(system_message=system_message, messages=messages)
