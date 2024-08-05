import argparse
import os
import importlib
import yaml
from pygentic.llm_backends import GenerationSpec, LlamaCpp
from pygentic import FileOutputDevice, Agent, FileLoadingConfig
from pygentic import run_agent
from pygentic.loaders import get_default_loaders


def build_llamacpp(spec):
    base_url = spec['base_url']
    sampling_config = spec['sampling_config']
    stop_token = spec['stop_token']
    proxies = spec.get('proxies', None)

    generation_spec = GenerationSpec(sampling_config=sampling_config,
                                     stop_word=stop_token)

    return LlamaCpp(base_url, generation_spec, proxies=proxies)


llm_builders = {
    'llama.cpp': build_llamacpp
}


def build_llms(spec):
    llms = {}
    for name, llm_spec in spec.get('llms', {}).items():
        backend = llm_spec.get('backend')

        supported_backends = list(llm_builders.keys())

        if not backend:
            raise ValueError(f'"{backend}" must be provided')
        if backend not in supported_backends:
            raise ValueError(f'Backend "{backend}" is not supported yet. '
                             f'Supported backends are {supported_backends}')

        builder = llm_builders[backend]
        llms[name] = builder(llm_spec)

    return llms


def import_tool(path):
    parts = path.split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]

    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def load_tools(agent_spec):
    tools = agent_spec.get('tools', {})
    return {tool_name: import_tool(path) for tool_name, path in tools.items()}


def load_prompt(spec):
    prompt_file = spec.get('prompt', "")

    if not os.path.isfile(prompt_file):
        return ""
    
    with open(prompt_file) as f:
        return f.read()


def build_agents(spec, llms):
    agents = {}
    for agent_name, agent_spec in spec.get('agents', {}).items():
        tools = load_tools(agent_spec)
        #done_tool = agent_spec.get('done_tool')
        system_message = load_prompt(agent_spec)
        max_rounds = agent_spec.get('max_rounds', 5)
        log_file = agent_spec.get('log_file')
        output_device = FileOutputDevice(log_file) if log_file else None
        
        llm_name = agent_spec.get('llm')
        if not llm_name:
            raise ValueError(f"Coudn't find LLM configuration in YAML specification: missing key '{llm}'")

        if llm_name not in llms:
            raise ValueError(f'LLM "{llm_name}" not found in the YAML specification')
        llm = llms[llm_name]

        agent = Agent(llm, tools, system_message=system_message, max_rounds=max_rounds, output_device=output_device)
        agents[agent_name] = agent
    return agents


def get_loading_conf(spec):
    """Override default loaders by custom ones when provided"""
    loaders = get_default_loaders()

    loading_section = spec.get("file_loading", {})

    ignore_list = loading_section.get("ignore_list", [])
    stop_on_error = loading_section.get("stop_on_error", True)

    for extension, loader_str in loading_section.get("loaders", {}).items():
        try:
            module = importlib.import_module('pygentic.loaders')
            
            loader = getattr(module, loader_str)
        except AttributeError:
            parts = loader_str.split('.')
            loader_name = parts[-1]
            module_path = '.'.join(parts[:-1])
            module = importlib.import_module(module_path)
            loader = getattr(module, loader_name)
        
        loaders[extension] = loader
    
    return FileLoadingConfig(loaders, ignore_list, stop_on_error)


def connect_agents(spec, agents):
    for agent_name, agent_spec in spec.get('agents', {}).items():
        for name, sub_agent_name in agent_spec.get('sub_agents', {}).items():
            sub_agent = agents[sub_agent_name]
            agents[agent_name].add_subagent(name, sub_agent)


def load_yaml_spec(yaml_file_path):
    with open(yaml_file_path, 'r') as f:
        spec = yaml.safe_load(f)

    loading_config = get_loading_conf(spec)
    llms = build_llms(spec)
    agents = build_agents(spec, llms)
    connect_agents(spec, agents)

    for agent in agents.values():
        agent.set_loading_config(loading_config)

    entrypoint = spec.get('entrypoint')
    if entrypoint is None:
        raise ValueError("Entry point '{entrypoint}' key not found in the YAML specification")

    entrypoint = spec['entrypoint']
    try:
        main_agent_name = entrypoint['agent']
        inputs = entrypoint['inputs']
    except KeyError:
        raise ValueError("Entry point must to contain 'agent' and 'inputs' keys")

    main_agent = agents.get(main_agent_name)
    if not main_agent:
        raise ValueError(f"Entry point agent named '{main_agent_name}' not found")
    
    files = []
    for file_entry in entrypoint.get('files', []):
        entry = {}
        entry['path'] = file_entry['path']
        if 'loader' in file_entry:
            entry['loader'] = file_entry['loader']
        files.append(entry)
    
    budgets = spec.get('budgets', {})
    
    budgets_dict = dict(max_eval = budgets.get("max_eval", 90000),
                        max_gen = budgets.get("max_gen", 90000),
                        max_total = budgets.get("max_total", 90000))
    return main_agent, inputs, files, budgets_dict


def main(yaml_file_path):
    main_agent, inputs, files, budgets_dict = load_yaml_spec(yaml_file_path)
    return run_agent(main_agent, inputs, files, **budgets_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs agents using the given yaml specification')
    parser.add_argument('yaml_file', help='Path to the yaml file containing agent specifications')
    args = parser.parse_args()

    yaml_file_path = args.yaml_file
    result = main(yaml_file_path)
    print(f'Program successfully finished with result: {result}')
