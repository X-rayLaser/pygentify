from dataclasses import dataclass


@dataclass
class Message:
    sections: list

    @classmethod
    def text_message(cls, text):
        section = TextSection(text)
        return cls([section])

    def __str__(self):
        return ''.join(str(section) for section in self.sections)


class Section:
    def __str__(self):
        raise NotImplementedError


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
