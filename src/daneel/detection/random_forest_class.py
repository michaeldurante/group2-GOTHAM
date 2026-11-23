'''Copiato al volo solo per risolvere il problem path di unput, poi ci lavoro con calma'''
from pathlib import Path
import yaml
import pandas as pd


class RandomForestDetection:
    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path).resolve()
        self.yaml_dir = self.yaml_path.parent

        with open(self.yaml_path, "r") as f:
            config = yaml.safe_load(f)

        self.params = config.get("detection", config)
        self.dataset_path = (self.yaml_dir / self.params["dataset_path"]).resolve()

        print(self.dataset_path)

    # def load_data(self):
    #     df = pd.read_csv(self.dataset_path)
    #     # ... preprocessing, split X_train/X_test e y_train/y_test come nel notebook ...
    #     return df


    def test(self):
        print('funziona')
        
path = '../compu_astro/group2-GOTHAM/rf_hyperparams.yaml'
model = RandomForestDetection(path)
