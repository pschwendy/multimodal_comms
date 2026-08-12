import importlib.util

def load_class_from_file(file_path, class_name):
    # Dynamically load the module
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the class by name
    cls = getattr(module, class_name)
    
    return cls

def load_classes_from_config(config):
    # Load Expert class
    expert_configs = config['Experts']
    ExpertClasses = [load_class_from_file(expert_config['file_path'], expert_config['class_name']) for expert_config in expert_configs]
    
    # Load Solver class
    solver_configs = config['Solvers']
    SolverClasses = [load_class_from_file(solver_config['file_path'], solver_config['class_name']) for solver_config in solver_configs]
    
    # Return the class objects so they can be instantiated later
    return ExpertClasses, SolverClasses, config

def load_one_agent_from_config(config):
    # Load Expert class
    agent_config = config['Agent']
    AgentClass = load_class_from_file(agent_config['file_path'], agent_config['class_name'])
    
    # Return the class objects so they can be instantiated later
    return AgentClass, config