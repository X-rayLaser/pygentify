from jinja2 import Environment, PackageLoader, select_autoescape
env = Environment(
    loader=PackageLoader("pygentic"),
    autoescape=select_autoescape()
)