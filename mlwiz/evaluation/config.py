import json


class Config:
    r"""
    Simple class to manage the configuration dictionary as a Python object
    with fields.

    Args:
        config_dict (dict): the configuration dictionary
    """

    def __init__(self, config_dict: dict):
        """
        Initialize the configuration wrapper.

        Args:
            config_dict (dict): Configuration dictionary to wrap.

        Side effects:
            Stores the dictionary on the instance as ``config_dict``.
        """
        self.config_dict = config_dict

    def __getattr__(self, attr: str):
        """
        Return the value associated with ``attr`` in the wrapped dictionary.

        Args:
            attr (str): Dictionary key to look up.

        Returns:
            object: Value stored under ``attr``.

        Raises:
            KeyError: If ``attr`` is not present in the dictionary.
        """
        return self.config_dict[attr]

    def __getitem__(self, item: str):
        """
        Return the value associated with ``item`` in the wrapped dictionary.

        Args:
            item (str): Dictionary key to look up.

        Returns:
            object: Value stored under ``item``.

        Raises:
            KeyError: If ``item`` is not present in the dictionary.
        """
        return self.config_dict[item]

    def __contains__(self, item: str) -> bool:
        """
        Return whether the wrapped dictionary contains ``item``.

        Args:
            item (str): Key to test.

        Returns:
            bool: ``True`` if present, ``False`` otherwise.
        """
        return item in self.config_dict

    def __len__(self) -> int:
        """
        Return the number of keys in the wrapped dictionary.
        """
        return len(self.config_dict)

    def __iter__(self):
        """
        Iterate over keys in the wrapped dictionary.

        Returns:
            Iterator[str]: Iterator over dictionary keys.
        """
        return iter(self.config_dict)

    def get(self, key: str, default: object = None) -> object:
        """
        Returns the key from the dictionary if present, otherwise the default
        value specified

        Args:
            key (str): the key to look up in the dictionary
            default (`object`): the default object

        Returns:
            a value from the dictionary
        """
        return self.config_dict.get(key, default)

    def keys(self) -> set:
        r"""
        Return a view on the configuration keys.

        Returns:
            dict_keys: View over keys in the dictionary.
        """
        return self.config_dict.keys()

    def items(self) -> list:
        r"""
        Return a view on (key, value) pairs.

        Returns:
            dict_items: View over (key, value) pairs.
        """
        return self.config_dict.items()

    def __str__(self) -> str:
        """
        Return an indented JSON representation of the configuration.

        Returns:
            str: Pretty-printed JSON string.
        """
        return json.dumps(self.config_dict, sort_keys=True, indent=4)
