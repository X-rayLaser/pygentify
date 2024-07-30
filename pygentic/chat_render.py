LLAMA3_START_HEADER_ID = "<|start_header_id|>";
LLAMA3_END_HEADER_ID = "<|end_header_id|>";
LLAMA3_EOT_ID = "<|eot_id|>";


def llamaRoleTemplate(role):
    return f'{LLAMA3_START_HEADER_ID}{role}{LLAMA3_END_HEADER_ID}\n\n%message{LLAMA3_EOT_ID}'


llama3_template = {
    'question': llamaRoleTemplate("user"),
    'answer': llamaRoleTemplate("assistant"),
    'systemMessage': llamaRoleTemplate("system"),
    'startOfText': "<|begin_of_text|>",
    'promptSuffix': f'{LLAMA3_START_HEADER_ID}assistant{LLAMA3_END_HEADER_ID}\n\n',
    'continuationPrefix': f'{LLAMA3_EOT_ID}{LLAMA3_START_HEADER_ID}assistant{LLAMA3_END_HEADER_ID}\n\n'
}

default_template = llama3_template


class ChatRenderer:
    def __init__(self, template_spec, use_bos=False):
        self.spec = template_spec
        self.use_bos = use_bos

    def __call__(self, system_message, messages):
        raise NotImplementedError


class ChatRendererToString(ChatRenderer):
    def __call__(self, system_message, messages):
        questionTemplate = self.spec['question']
        answerTemplate = self.spec['answer']

        conversation = ''

        system_template = self.spec['systemMessage']

        if system_message:
            conversation += system_template.replace('%message', str(system_message))

        for i, msg in enumerate(messages):
            template = questionTemplate if i % 2 == 0 else answerTemplate
            if hasattr(msg, 'sections'):
                text = str(msg)
            elif hasattr(msg, 'text'):
                text = msg.text
            elif 'text' in msg:
                text = msg['text']
            elif isinstance(msg, str):
                text = msg
            else:
                raise Exception("Cannot handle object", msg)

            conversation += template.replace('%message', text)

        conversation = conversation + self.spec['promptSuffix']

        if self.use_bos:
            conversation = self.spec['startOfText'] + conversation
        return conversation
