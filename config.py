import json
import os

CONFIG_FILE = 'config.json'

class Config:
    def __init__(self, config_file=CONFIG_FILE):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_file):
            default_config = {
                "rebuild_model": True,
                "target_stock_code": "000001;600000",
                "backtest_year": 3,
                "stop_loss_threshold": 0.10,
                "stop_profit_threshold": 0.10,
                "model_path": "./models",
                "init_capital": 1000000.0,
                "max_hold_days": 20
            }
            self._save_config_dict(default_config)
            return default_config
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_config_dict(self, config_dict):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self._save_config_dict(self.config)

    def get_all(self):
        return self.config

    def set_all(self, new_config):
        self.config.update(new_config)
        self._save_config_dict(self.config)

# Global config instance for easier access
cfg = Config()
