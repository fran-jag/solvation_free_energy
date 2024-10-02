"""
Model for application setup configuration.
"""

from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """
    ML Settings class that contains configuration variables
    """

    model_config = SettingsConfigDict(env_file='./config/.env',
                                      env_file_encoding='utf-8',
                                      extra='ignore',
                                      )
    # Remove warning caused by model_ variable in pydantic
    model_config['protected_namespaces'] = ('settings_',)

    model_path: DirectoryPath
    data_path: FilePath
    model_name: str


# Initialize settings
model_settings = ModelSettings()
