from importlib import import_module

def dynamic_import_function(function_path):
    """
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    """
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
