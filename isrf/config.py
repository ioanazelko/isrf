# isrf/config.py
import os
import json
from pathlib import Path

class ISRFConfig:
    def __init__(self):
        self.config_file = Path.home() / '.isrf' / 'config.json'
        self.config = self._load_config()
    
    def _load_config(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def set_data_path(self, data_type, path):
        """Set path for specific data type ('stars' or 'dust')"""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        self.config[f'{data_type}_data_path'] = str(path)
        self._save_config()
    
    def get_data_path(self, data_type):
        """Get configured path for data type"""
        key = f'{data_type}_data_path'
        if key in self.config:
            return Path(self.config[key])
        return None
    
    def _save_config(self):
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
