import argparse
import os
import importlib
import yaml
from pygentic.llm_backends import GenerationSpec, LlamaCpp
from pygentic import FileOutputDevice, Agent, get_default_loaders, FileLoadingConfig


def build_llms(spec):
    llms = {}
    for name, llm_spec in spec.get('llms', {}).items():
        llm_host = llm_spec['host']
        llm_port = llm_spec['port']
        inference_config = llm_spec['inference_config']
        sampling_config = llm_spec['sampling_config']
        stop_token = llm_spec['stop_token']
        generation_spec = GenerationSpec(inference_config=inference_config,
                                         sampling_config=sampling_config,
                                         stop_word=stop_token)

        proxies = llm_spec.get('proxies', None)
        llms[name] = LlamaCpp(llm_host, llm_port, generation_spec, proxies=proxies)

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

    loading_config = get_loading_conf()
    llms = build_llms(spec)
    agents = build_agents(spec, llms)
    connect_agents(spec, agents)

    for agent in agents:
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
    return main_agent, inputs, files


def main(yaml_file_path):
    main_agent, inputs, files = load_yaml_spec(yaml_file_path)
    return main_agent(inputs, files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs agents using the given yaml specification')
    parser.add_argument('yaml_file', help='Path to the yaml file containing agent specifications')
    args = parser.parse_args()

    yaml_file_path = args.yaml_file
    result = main(yaml_file_path)
    print(f'Program successfully finished with result: {result}')
