
def _registry_generic(module_dict, module_name, module):
    assert module_name not in module_dict 
    module_dict[module_name] = module 


class Registry(dict):

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__()

    def registry(self, module_name, module=None):
        if module is not None:
            _registry_generic(self, module_name, module)
            return 
        
        def register_fn(fn):
            _registry_generic(self, module_name, fn)
            return fn 
        return register_fn