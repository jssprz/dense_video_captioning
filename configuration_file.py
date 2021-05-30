from configparser import ConfigParser


class ConfigurationFile:
    def __init__(self, config_path, section_name):
        self.__config = ConfigParser()
        self.__config.read(config_path)

        try:
            section = self.__config[section_name]
        except Exception:
            raise ValueError(" {} is not a valid section".format(section_name))

        self.__data_dir = section["DATA_DIR"]

    @property
    def data_dir(self):
        return self.__data_dir
