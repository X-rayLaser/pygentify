import re
import os
from dataclasses import dataclass
from pypdf import PdfReader

from .misc import TextSection


def get_default_loaders(message_factory):
    # todo: support more extensions
    loaders = {}
    text_extensions = [".txt", ".text", ".py", ".c", ".h", ".cpp", "hpp", ".rb", ".csv"]
    for ext in text_extensions:
        loaders[ext] = PlainTextLoader(message_factory)

    loaders[".pdf"] = SimplePdfLoader(message_factory)
    return loaders


class FileLoader:
    start_of_file = '\n{}\n'
    end_of_file = 'EOF\n'

    def __init__(self, message_factory):
        self.message_factory = message_factory

    def __call__(self, path):
        messages = []

        text = self.process_file(path)

        sof_msg = self.message_factory.create_user_msg(self.start_of_file.format(path))
        eof_msg = self.message_factory.create_user_msg(self.end_of_file.format(path))

        messages.append(sof_msg)
        messages.append(self.message_factory.create_user_msg(text))
        messages.append(eof_msg)
        return messages

    def process_file(self, path):
        raise NotImplementedError


class NullLoader(FileLoader):
    def process_file(self, path):
        raise LoaderError(f'Cannot find a suitable loader for "{path}"')


class PlainTextLoader(FileLoader):
    def process_file(self, path):
        with open(path) as f:
            return f.read()


class SimplePdfLoader(FileLoader):
    def process_file(self, path):
        reader = PdfReader(path)
        number_of_pages = len(reader.pages)
        content = ""

        for i in range(number_of_pages):
            page = reader.pages[i]
            content += page.extract_text()
        return content


@dataclass
class FileLoadingConfig:
    loaders: dict
    ignore_list: list
    stop_on_error: bool

    @classmethod
    def empty_config(cls):
        return FileLoadingConfig({}, [], True)


class FileTreeLoader:
    def __init__(self, config: FileLoadingConfig, message_factory):
        self.config = config
        self.message_factory = message_factory

    def __call__(self, path):
        try:
            return self._load(path)
        except Exception as e:
            if self.config.stop_on_error:
                raise LoaderError(f'Failed to load file "{path}": {str(e)}')
            return []

    def _load(self, path):
        for pattern in self.config.ignore_list:
            if re.match(pattern, path):
                # skipping this file or directory
                return []

        sections = []
        if os.path.isfile(path):
            _, extension = os.path.splitext(path)
            loader = self.config.loaders.get(extension.lower(), NullLoader(self.message_factory))
            sections = loader(path)
        else:
            for name in os.listdir(path):
                sub_path = os.path.join(path, name)
                loader = FileTreeLoader(self.config, self.message_factory)
                sections.extend(loader(sub_path))

        return sections


class LoaderError(Exception):
    pass
