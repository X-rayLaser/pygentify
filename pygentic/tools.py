import math
import json

from .tool_calling import register


@register()
def add(num1:float, num2:float) -> float:
    """Add two numbers and returns their sum
    
    num1: first summand
    num2: second summand
    """
    return num1 + num2

add.usage_examples = [{"num1": 23, "num2": 19}, {"num1": -32, "num2": 9}]

@register()
def subtract(num1, num2):
    return num1 - num2


@register()
def multiply(num1, num2):
    return num1 * num2


@register()
def divide(num1, num2):
    """
    Divides first argument by second argument
    
    The second argument must not be zero, otherwise ZeroDivisionError will be raised
    """
    return num1 / num2


divide.usage_examples = [{"num1": 23, "num2": 12}, {"num1": 23, "num2": 1}, {"num1": 23, "num2": 22}]


def round(x):
    return round(x)


def sqrt(number):
    return math.sqrt(number)


def pow(num1, num2):
    return math.pow(num1, num2)


def sin(rads):
    return math.sin(rads)


def cos(rads):
    return math.cos(rads)


cos.usage_examples = ["cos(0.2)"]


class SearchProvider:
    def search(self, query, **kwargs):
        return []


def get_web_search(provider):
    def search(query, **kwargs):
        try:
            results = provider.search(query, **kwargs)
            return json.dumps(results)
        except Exception:
            print("Exception")
            raise
    return search
