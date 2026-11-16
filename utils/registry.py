class Registry:
    """简单的名字 -> 类/函数 注册表，供 datasets / methods 扩展使用。"""

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, name=None):
        def deco(obj):
            key = name or obj.__name__
            if key in self._module_dict:
                raise KeyError(f"'{key}' 已在 {self._name} 注册表中注册")
            self._module_dict[key] = obj
            return obj
        return deco

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(
                f"'{name}' 不在 {self._name} 注册表中，可用: {sorted(self._module_dict)}"
            )
        return self._module_dict[name]
