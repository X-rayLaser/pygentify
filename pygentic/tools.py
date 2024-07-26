import math
import json


def add(num1, num2):
    return num1 + num2


def subtract(num1, num2):
    return num1 - num2


def multiply(num1, num2):
    return num1 * num2


def divide(num1, num2):
    return num1 / num2


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
