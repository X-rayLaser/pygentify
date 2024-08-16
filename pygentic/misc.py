from dataclasses import dataclass


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
