import re
import os
from dataclasses import dataclass
from .misc import TextSection


def get_default_loaders():
    loaders = {}
    text_extensions = [".txt", ".text", ".py", ".c", ".h", ".cpp", "hpp", ".rb", ".csv"]
    for ext in text_extensions:
        loaders[ext] = PlainTextLoader()

    return loaders


class FileLoader:
    def __call__(self, path):
        raise NotImplementedError


class NullLoader(FileLoader):
    def __call__(self, path):
        raise LoaderError(f'Cannot find a suitable loader for "{path}"')


class PlainTextLoader(FileLoader):
    def __call__(self, path):
        with open(path) as f:
            text = f.read()
        
        sections = []
        start_of_file = TextSection(f'\n{path}\n')
        end_of_file = TextSection(f'EOF\n')
        sections.append(start_of_file)
        sections.append(TextSection(text))
        sections.append(end_of_file)
        return sections


@dataclass
class FileLoadingConfig:
    loaders: dict
    ignore_list: list
    stop_on_error: bool

    @classmethod
    def empty_config(cls):
        return FileLoadingConfig({}, [], True)


class FileTreeLoader:
    def __init__(self, config: FileLoadingConfig):
        self.config = config

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
            loader = self.config.loaders.get(extension.lower(), NullLoader())
            sections = loader(path)
        else:
            for name in os.listdir(path):
                sub_path = os.path.join(path, name)
                loader = FileTreeLoader(self.config)
                sections.extend(loader(sub_path))

        return sections


class LoaderError(Exception):
    pass
