from dataclasses import dataclass
from copy import deepcopy
import yaml


@dataclass
class Message:
    sections: list
    role: str

    @classmethod
    def text_message(cls, text, role):
        section = TextSection(text)
        return cls([section], role)

    def __str__(self):
        return '\n'.join(str(section) for section in self.sections)

    def clone(self):
        sections = [TextSection(str(s)) for s in self.sections]
        return Message(sections, self.role)


class Section:
    def __str__(self):
        raise NotImplementedError


@dataclass
class ToolCallSection(Section):
    name: str
    arg_dict: dict

    def __str__(self):
        return self.name


@dataclass
class ResultSection(Section):
    name: str
    content: str

    def __str__(self):
        return self.name


@dataclass
class TextSection(Section):
    content: str

    def __str__(self):
        return self.content


@dataclass
class ImageSection(Section):
    content: bytes

    def __str__(self):
        return f'image of {len(self.content)} bytes length'


def override_dict(base_dict, new_dict):
    res = deepcopy(base_dict)
    
    for k, v in dict(new_dict).items():
        if k in res:
            old_value = res[k]
            res[k] = override_structure(old_value, v)
        else:
            res[k] = v

    return res


def override_structure(base_value, new_value):
    if is_primitive(new_value):
        return new_value

    if is_primitive(base_value):
        return new_value

    if type(base_value) != type(new_value):
        return new_value

    # here we know that both are collections of the same type

    if isinstance(base_value, list):
        return new_value

    return override_dict(base_value, new_value)


def is_primitive(x):
    return not (isinstance(x, dict) or isinstance(x, list) or isinstance(x, bytes))


def finalize(x):
    if is_primitive(x):
        return x

    if isinstance(x, list):
        return [finalize(item) for item in x]

    mapping = deepcopy(x)

    for k, v in dict(mapping).items():
        mapping[k] = finalize(v)

    inheritance_path = mapping.get('inherit')

    if inheritance_path:
        del mapping['inherit']
        base_dict = load_yaml(inheritance_path)
        mapping = override_structure(base_dict, mapping)
    return mapping


def load_yaml(path):
    with open(path) as f:
        spec = yaml.safe_load(f)

    return finalize(spec)
