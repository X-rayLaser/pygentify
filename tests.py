import unittest
import json
from unittest.mock import Mock
from pygentic import (
    ToolUseNotFoundError, find_tool_use, parse_tool_use, render_tool_use_string,
    Agent, BaseLLM, BadToolUseError, TooManyRoundsError, ActionDispatcher, 
    ToolUseFailedError, contains_tool_use, render_tool_use_error, ChatRendererToString,
    ToolDoesNotExistError
)


class MockLLM(BaseLLM):
    def __init__(self, response):
        self.response = response

    def __call__(self, text):
        return self.response


class FindToolUseTests(unittest.TestCase):
    def test(self):
        self.assertRaises(ToolUseNotFoundError, find_tool_use, "")
        self.assertRaises(ToolUseNotFoundError, find_tool_use, "  ")
        self.assertRaises(ToolUseNotFoundError, find_tool_use, "some normal text")
        
        tool_use_str = '<|tool_use_start|>whatever<|tool_use_end|>'
        offset, length, body = find_tool_use(tool_use_str)
        self.assertEqual(offset, 0)
        self.assertEqual(length, len(tool_use_str))

        offset, length, body = find_tool_use(f'__{tool_use_str}')
        self.assertEqual(offset, 2)
        self.assertEqual(length, len(tool_use_str))
        self.assertEqual(body, "whatever")

    def test_multiple_matches(self):
        tool_use_str = '<|tool_use_start|>whatever<|tool_use_end|>'
        tool_use_str += ' <|tool_use_start|>another<|tool_use_end|>'

        offset, length, body = find_tool_use(tool_use_str)
        self.assertEqual(offset, 0)
        self.assertEqual(length, len('<|tool_use_start|>whatever<|tool_use_end|>'))
        self.assertEqual(body, "whatever")

    def test_edge_cases(self):
        tool_use_str = '<|tool_use_start|>'
        with self.assertRaises(ToolUseNotFoundError):
            find_tool_use(tool_use_str)

        tool_use_str = '|tool_use_end|>'
        with self.assertRaises(ToolUseNotFoundError):
            find_tool_use(tool_use_str)

        tool_use_str = '<|tool_use_start||tool_use_end|>'
        with self.assertRaises(ToolUseNotFoundError):
            find_tool_use(tool_use_str)

    def test_whitespace(self):
        tool_use_str = '   <|tool_use_start|>whatever<|tool_use_end|>'
        offset, length, body = find_tool_use(tool_use_str)
        self.assertEqual(offset, 3)
        self.assertEqual(length, len(tool_use_str.strip()))
        self.assertEqual(body, "whatever")

    def test_characters(self):
        tool_use_str = '<|tool_use_start|>hello 123 [abc] {def} <|tool_use_end|>'
        offset, length, body = find_tool_use(tool_use_str)
        self.assertEqual(offset, 0)
        self.assertEqual(length, len(tool_use_str))
        self.assertEqual(body, "hello 123 [abc] {def} ")

    def test_quotes(self):
        tool_use_str = '<|tool_use_start|>"hello" <|tool_use_end|>'
        offset, length, body = find_tool_use(tool_use_str)
        self.assertEqual(offset, 0)
        self.assertEqual(length, len(tool_use_str))
        
        self.assertEqual(body, '"hello" ')

    def test_apostrophes(self):
        tool_use_str = "<|tool_use_start|>'hello's' <|tool_use_end|>"
        offset, length, body = find_tool_use(tool_use_str)
        self.assertEqual(offset, 0)
        self.assertEqual(length, len(tool_use_str))
        self.assertEqual(body, "'hello's' ")

    def test_brackets(self):
        tool_use_str = '<|tool_use_start|>[abc] {def} <|tool_use_end|>'
        offset, length, body = find_tool_use(tool_use_str)
        self.assertEqual(offset, 0)
        self.assertEqual(length, len(tool_use_str))
        self.assertEqual(body, "[abc] {def} ")


class TestParseToolUse(unittest.TestCase):

    def test_valid_json(self):
        text = '{"tool_name": "my_tool", "args": {"arg1": "value1", "arg2": "value2"}}'
        tool_name, args = parse_tool_use(text)
        self.assertEqual(tool_name, "my_tool")
        self.assertEqual(args, {"arg1": "value1", "arg2": "value2"})

    def test_missing_tool_name(self):
        text = '{"args": {"arg1": "value1", "arg2": "value2"}}'
        with self.assertRaises(ValueError):
            parse_tool_use(text)

    def test_invalid_json(self):
        text = 'not a valid json string'
        with self.assertRaises(ValueError):
            parse_tool_use(text)

    def test_empty_string(self):
        text = ''
        with self.assertRaises(ValueError):
            parse_tool_use(text)

    def test_integer_arg(self):
        text = '{"tool_name": "my_tool", "args": {"arg1": 123}}'
        tool_name, args = parse_tool_use(text)
        self.assertEqual(tool_name, "my_tool")
        self.assertEqual(args, {"arg1": 123})

    def test_float_arg(self):
        text = '{"tool_name": "my_tool", "args": {"arg1": 3.14}}'
        tool_name, args = parse_tool_use(text)
        self.assertEqual(tool_name, "my_tool")
        self.assertEqual(args, {"arg1": 3.14})

    def test_boolean_arg(self):
        text = '{"tool_name": "my_tool", "args": {"arg1": true}}'
        tool_name, args = parse_tool_use(text)
        self.assertEqual(tool_name, "my_tool")
        self.assertEqual(args, {"arg1": True})

    def test_find_and_parse_tool_use_with_valid_payload(self):
        tool_use_str = '<|tool_use_start|>{"tool_name": "my_tool", "args": {"arg1": 123}}<|tool_use_end|>'
        offset, length, body = find_tool_use(tool_use_str)
        name, args = parse_tool_use(body)
        self.assertEqual(name, 'my_tool')
        self.assertEqual(args, {'arg1': 123})

    def test_find_and_parse_tool_use_with_invalid_payload(self):
        tool_use_str = '<|tool_use_start|>not a valid json string<|tool_use_end|>'
        offset, length, body = find_tool_use(tool_use_str)
        with self.assertRaises(ValueError):
            parse_tool_use(body)


class RederToolUseString(unittest.TestCase):
    def test_render_tool_use_string_with_valid_args(self):
        tool_name = "calculator"
        arg_dict = {"op1": 4, "op2": 6, "operation": "+"}
        result = "10"
        expected_output = f'{{"tool_name": "{tool_name}", "args": {{"op1": 4, "op2": 6, "operation": "+"}}, "result": "{result}"}}'
        self.assertEqual(render_tool_use_string(tool_name, arg_dict, result), expected_output)

    def test_render_tool_use_string_with_empty_result(self):
        tool_name = "calculator"
        arg_dict = {"op1": 4, "op2": 6, "operation": "+"}
        result = ""
        expected_output = f'{{"tool_name": "{tool_name}", "args": {{"op1": 4, "op2": 6, "operation": "+"}}}}'
        self.assertEqual(render_tool_use_string(tool_name, arg_dict, result), expected_output)

    def test_render_tool_use_string_with_missing_args(self):
        tool_name = "calculator"
        arg_dict = {}
        result = "10"

        expected_output = f'{{"tool_name": "{tool_name}", "args": {{}}, "result": "{result}"}}'
        self.assertEqual(render_tool_use_string(tool_name, arg_dict, result), expected_output)


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = Agent(llm=MockLLM(''), tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, max_rounds=5)

    def test_invalid_tool_use_syntax(self):
        mock_llm = MockLLM('Invalid tool use syntax')
        agent = Agent(llm=mock_llm, tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, max_rounds=5)
        with self.assertRaises(TooManyRoundsError):
            agent({'input': ''})

    def test_tool_not_found(self):
        mock_llm = MockLLM('<|tool_use_start|>{"tool_name": "non_existent_tool", "args": {}}<|tool_use_end|>')
        agent = Agent(llm=mock_llm, tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, max_rounds=5)
        with self.assertRaises(TooManyRoundsError):
            agent({'input': ''})

    def test_malformed_json(self):
        mock_llm = MockLLM('<|tool_use_start|>{"tool_name": "my_tool", "args": [123<|tool_use_end|>')
        agent = Agent(llm=mock_llm, tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, max_rounds=5)
        with self.assertRaises(TooManyRoundsError):
            agent({'input': ''})

    def test_done_tool_no_args(self):
        mock_llm = MockLLM('<|tool_use_start|>{"tool_name": "done_tool", "args": {}}<|tool_use_end|>')
        done_tool = lambda: ""
        agent = Agent(llm=mock_llm, tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, done_tool=done_tool)
        result = agent({'input': ''})
        self.assertEqual(result, "")

    def test_done_tool_one_arg(self):
        mock_llm = MockLLM('<|tool_use_start|>{"tool_name": "done_tool", "args": {"arg1": 123}}<|tool_use_end|>')
        done_tool = lambda arg1: f"result with arg1={arg1}"
        agent = Agent(llm=mock_llm, tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, done_tool=done_tool)
        result = agent({'input': ''})
        self.assertEqual(result, "result with arg1=123")

    def test_done_tool_two_args(self):
        mock_llm = MockLLM('<|tool_use_start|>{"tool_name": "done_tool", "args": {"arg1": 123, "arg2": "abc"}}<|tool_use_end|>')
        done_tool = lambda arg1, arg2: f"result with args {arg1} and {arg2}"
        agent = Agent(llm=mock_llm, tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, done_tool=done_tool)
        result = agent({'input': ''})
        self.assertEqual(result, "result with args 123 and abc")

    def test_delegate_to_subagent(self):
        subagent_inputs = []
        class Subagent:
            def __call__(self, **kwargs):
                subagent_inputs.append(kwargs)
                return "subagent response"

        llm = Mock()
        payload = {"tool_name": "delegate", "args": {"name": "subagent", "inputs": {"input": "some input"}}}

        llm.return_value = f'<|tool_use_start|>{json.dumps(payload)}<|tool_use_end|>'
        agent = Agent(llm, {"tool1": lambda: "tool1 response"}, max_rounds=1)

        agent.add_subagent("subagent", Subagent())

        with self.assertRaises(TooManyRoundsError):
            result = agent({"input": "some input"})

        self.assertEqual(subagent_inputs, [{"input": "some input"}])

    def test_too_many_rounds_error(self):
        agent = Agent(llm=MockLLM('<|tool_use_start|>{"action_name": "use_tool", "tool_name": "tool1", "args": {}}<|tool_use_end|>'),
                      tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, 
                      max_rounds=5)

        with self.assertRaises(TooManyRoundsError):
            agent({'input': ''})

    def test_subagent_ask_parent_agent_question(self):
        self.agent = Agent(llm=MockLLM(''),
                           tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'},
                           max_rounds=5)
        self.subagent = Agent(llm=MockLLM('<|tool_use_start|>{"tool_name": "clarify", "args": {"text": "some text"}}<|tool_use_end|>'), 
                              tools={'tool1': lambda: 'result1', 'tool2': lambda: 'result2'}, max_rounds=1)

        self.agent.ask_question = Mock()
        self.agent.ask_question.return_value = "response"
        self.subagent.parent = self.agent
        with self.assertRaises(TooManyRoundsError):
            self.subagent({'input': ''})
        self.assertEqual(("some text", ), self.agent.ask_question.call_args.args)

        # todo: test what send_message is doing


class TestActionDispatcher(unittest.TestCase):

    def setUp(self):
        class Foo:
            tools = {}

        self.agent = Foo()  # Replace with a real agent instance if needed
        self.action_handlers = {'action1': lambda x, y: y, 'action2': lambda x, y: (x + y, y)}

    def test_call_with_handler(self):
        dispatcher = ActionDispatcher(self.agent, self.action_handlers)
        result = dispatcher('action1', {'x': 1, 'y': 2})
        self.assertEqual(result, {'x': 1, 'y': 2})

    def test_call_without_handler(self):
        dispatcher = ActionDispatcher(self.agent, self.action_handlers)
        with self.assertRaises(ToolDoesNotExistError):
            dispatcher('non_existent_action', {})

    def test_call_with_tool(self):
        class Tool:
            def __init__(self, x, y): pass
            def __call__(self, **kwargs): return kwargs

        self.agent.tools = {'tool': Tool(1, 2)}
        dispatcher = ActionDispatcher(self.agent, self.action_handlers)
        result = dispatcher('tool', {'x': 3, 'y': 4})
        self.assertEqual(result, {'x': 3, 'y': 4})

    def test_call_with_non_existing_tool(self):
        self.agent.tools = {'tool': lambda x, y: (x, y)}
        dispatcher = ActionDispatcher(self.agent, self.action_handlers)
        with self.assertRaises(ToolDoesNotExistError):
            dispatcher('non_existent_tool', {'x': 1, 'y': 2})

    def test_call_with_failure_action(self):
        dispatcher = ActionDispatcher(self.agent, self.action_handlers)
        with self.assertRaises(ToolUseFailedError):
            dispatcher('failure', {})


class TestContainsToolUse(unittest.TestCase):

    def setUp(self):
        self.s1 = "<|tool_use_start|>Hello<|tool_use_end|>"
        self.s2 = "No tool use"
        self.s3 = "<|tool_use_start|>World<|tool_use_end|>"

    def test_contains_tool_use_true(self):
        self.assertTrue(contains_tool_use(self.s1))
        self.assertTrue(contains_tool_use(self.s3))

    def test_contains_tool_use_false(self):
        self.assertFalse(contains_tool_use(self.s2))


class TestRenderToolUseError(unittest.TestCase):

    def test_render_tool_use_error(self):
        tool_name = "My Tool"
        arg_dict = {"x": 1, "y": 2}
        error = "An error occurred"
        result = render_tool_use_error(tool_name, arg_dict, error)
        self.assertIsInstance(result, str)

    def test_render_tool_use_error_without_error(self):
        tool_name = "My Tool"
        arg_dict = {"x": 1, "y": 2}
        result = render_tool_use_error(tool_name, arg_dict)
        expected_result = '{"tool_name": "%s", "args": {"x": 1, "y": 2}}' % tool_name
        self.assertEqual(json.loads(result), json.loads(expected_result))

    def test_render_tool_use_error_with_error(self):
        tool_name = "My Tool"
        arg_dict = {"x": 1, "y": 2}
        error = "An error occurred"
        result = render_tool_use_error(tool_name, arg_dict, error)
        expected_result = '{"tool_name": "%s", "args": {"x": 1, "y": 2}, "error": "%s"}' % (tool_name, error)
        self.assertEqual(json.loads(result), json.loads(expected_result))


class TestChatRendererToString(unittest.TestCase):

    def setUp(self):
        self.template_spec = {
            'question': '[[%message]]',
            'answer': '(%message)',
            'systemMessage': '--%message--',
            'promptSuffix': '>>>',
            'continuationPrefix': ''
        }
        self.chat_renderer = ChatRendererToString(self.template_spec)

    def test_call_with_system_message_and_messages(self):
        system_message = "Hello, world!"
        messages = [{"text": "This is a message"}, {"text": "Another message"}]
        result = self.chat_renderer(system_message, messages)
        expected_result = f'--{system_message}--[[This is a message]](Another message)>>>'
        self.assertEqual(result, expected_result)

    def test_call_with_messages_only(self):
        messages = [{"text": "This is a message"}, {"text": "Another message"}]
        result = self.chat_renderer(None, messages)
        expected_result = f'[[This is a message]](Another message)>>>'
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
