import json
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from inspect import signature

import glori.settings.paths as paths
from glori.infra.logging import pretty_print_config
from glori.config.load import parse_preset


class modelConfig(object):
    """
    A class representing the configuration for a model. The purpose of this
    class is to provide a single, flexible object that can be passed to a model
    constructor, rather than passing a large number of arguments. This class stores
    the arguments needed to construct a model in a dictionary, but also as attributes,
    inspired by pandas DataFrame.

    Args:
        **kwargs: Keyword arguments representing the configuration parameters.

    Attributes:
        param_dict (dict): A dictionary containing the configuration parameters.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.param_dict = kwargs
        self.__dict__.update(self.param_dict)

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        Set an attribute value.

        Args:
            __name (str): The name of the attribute.
            __value (Any): The value to be set.
        """
        super().__setattr__(__name, __value)
        if __name != "param_dict":
            self.param_dict[__name] = __value

    def get(self, name: str, default: Any = None) -> Any:
        """
        Get an attribute value, with a default if the attribute does not exist.

        Args:
            name (str): The name of the attribute.
            default (Any): The default value to return if the attribute does not exist.
        Returns:
            Any: The value of the attribute, or the default value if the attribute does not exist
        """
        return getattr(self, name, default)

    @classmethod
    def from_preset(self, preset: Path | str) -> "modelConfig":
        config_file = parse_preset(preset, paths.MODEL_CONFIGS)
        return self(**json.loads(config_file.read_text()))

    def save_to_json(self, filepath: Path) -> None:
        """
        Save the configuration parameters to a JSON file.

        Args:
            filepath (Path): The path to the JSON file where the configuration will be saved.
        """
        with open(filepath, "w") as f:
            json.dump(self.param_dict, f, indent=4)

    def update(self, update_dict: dict) -> None:
        """
        Update the configuration parameters with a dictionary.

        Args:
            update_dict (dict): A dictionary containing the parameters to be updated.
        """
        self.param_dict.update(update_dict)
        self.__dict__.update(self.param_dict)

    def construct(self, cls: type, *args: Any, **kwargs: Any) -> Any:
        """
        Construct an object of a class using the configuration object.

        Args:
            cls (class): The class to be instantiated.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            An instance of the class constructed using the configuration object.
        """
        # Extract valid kwargs from hyperparams
        config_kwargs = {
            k: v
            for k, v in self.param_dict.items()
            if k in signature(cls).parameters.keys()
        }
        return cls(*args, **(kwargs | config_kwargs))

    def pretty_print(self) -> None:
        """
        Pretty print the configuration parameters.
        """
        pretty_print_config(self, title="Model Configuration")
