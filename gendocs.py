# todo: use LLM in order to convert loosely documented functions into proper docstrings in reStructured format
# and generate actual documentation in MarkDown
# load full module with tools into LLM context, keep asking to generate documentation for next function in a loop
# ask LLM to generate usage examples as well (correct and incorrect)
# save documentation in a file
# user must review docs and approve them by running gendocs.py approve; this will add doc files to index and compute
# hashes of actual function docstrings from which they were derived; when changes detected, only changed functions will
# participate in documentation regeneration

import os
import argparse
from inspect import signature, getsource
from pygentic.jinja_env import env
from pygentic.tool_calling import tool_registry
from pygentic.llm_backends import LlamaCpp, GenerationSpec
from pygentic.completion import TextCompleter
from pygentic import ChatHistory, default_template, Thread, TextSection, Message
from pygentic.tool_calling import default_tool_use_backend


def generate_docs(function_template, completer, output_dir):
    system_message = "Whatever"

    llama_template = default_template

    history = ChatHistory(system_message, llama_template)

    func_template = env.get_template(function_template)

    tool_use_helper = default_tool_use_backend()

    for name, func in tool_registry.items():
        src = getsource(func)

        prompt = "Create a high quality documentation for the following function:\n\n" + src
        history.add_message(prompt)
        input_text = history.full_text()

        doc_str = completer(input_text)

        history.add_message(func_doc)

        sig = signature(func)
        examples = []
        if hasattr(func, 'usage_examples'):
            for arg_dict in func.usage_examples:
                tool_use_str = tool_use_helper.render(name, arg_dict)
                examples.append(tool_use_str)

        func_doc = func_template.render(name=name, signature=str(sig), doctext=doc_str, usage_examples=examples)
        doc_file = os.path.join(output_dir, name + '.txt')
        with open(doc_file, "w") as f:
            f.write(doc_file)
        print(f'Saved documentation for "{name}" function to "{doc_file}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('llm_backend', type=str)
    parser.add_argument('base_url', type=str)
    parser.add_argument('--function_template', type=str, default="function_doc.jinja")

    parser.add_argument('--output_dir', type=str, default="docs")

    spec = GenerationSpec({}, stop_word=None)
    args = parser.parse_args()

    llm = LlamaCpp(args.base_url, spec, proxies={})
    completer = TextCompleter(llm)

    tool_use_helper = default_tool_use_backend()

    generate_docs(args.function_template, completer, args.output_dir)