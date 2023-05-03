import yaml


class ConfigFileModifier():    ### ou avec sklearn on fait le gridsearch classique ??? 
    def __init__(self, ) -> None: #list
        pass


    def modify_config_file():
        #on lit le fichier de config
        #on modifie les paramètres
        #on sauvegarde le fichier de config
        with open('config/expe.yml', 'rw') as stream:
            config = yaml.safe_load(stream)

    def ismoreconfig():
        pass


if __name__ == "__main__":
    config_file = "config/expe.yml"
    config = ConfigFileModifier()
    while config.list_conf:
        config.modify_config_file(config_file)

    expe_unit(config_file)