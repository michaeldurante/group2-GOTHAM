from pathlib import Path
from typing import Any

import yaml

class Parameters:
    """
    The parameters class, crucial to read configuration files and run a complex code

    Keyword arguments:
    input file -- path to the .yaml configuration file where daneel will extract all the important parameters
    Return: a Python dictionary with all the parameters contained in the input file
    """

    def __init__(self, input_file: str | Path):
        path = Path(input_file)
        yaml_dir = path.resolve().parent.parent
        if not path.is_file():
            raise FileNotFoundError(f"Parameter file not found: {path}")

        with path.open(encoding="utf-8") as in_f:
            loaded = yaml.safe_load(in_f)

        if loaded is None:
            self.params: dict[str, Any] = {}
        elif not isinstance(loaded, dict):
            raise ValueError("The parameter file must contain a YAML mapping")
        else:
            self.params = loaded

        '''
        create an absolute path to the csv file starting from the .yaml path
        create path to output directory where .yaml file is
        these paths do not depend on the user who is running daneel
        '''
        def update_paths(name):
            self.params = loaded.get(name, loaded)
            dataset_path = (yaml_dir / self.params['dataset_path']).resolve()
            output_dir = (yaml_dir / self.params['output_dir']).resolve()
            self.params.update({'dataset_path': dataset_path, 'output_dir': output_dir})
        
        if 'detection' in loaded:
            update_paths('detection')
        elif 'cnn' in loaded:
            update_paths('cnn')
        elif 'dream' in loaded:
            update_paths('dream')

        self._replace_none_strings(self.params)


    @staticmethod
    def _replace_none_strings(values: dict[str, Any]) -> None:
        """Convert legacy ``"None"`` values, including nested values."""
        for key, value in values.items():
            if value == "None":
                values[key] = None
            elif isinstance(value, dict):
                Parameters._replace_none_strings(value)


    def get(self, param: str, default: Any = None) -> Any:
        return self.params.get(param, default)
