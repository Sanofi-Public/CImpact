import yaml

def edit_config_file():
    """ Edit config file """
    file = "config.yaml"

    with open(file, 'r') as f:
        config = yaml.safe_load(f)
        login=input("Please enter your sanofi email:")
        config['login'] = login
    
    with open(file, "w") as ostream:
        yaml.dump(config, ostream, default_flow_style=False, sort_keys=False)


def generate_docs():
    pass

def main():
    edit_config_file()
    generate_docs()


if __name__ == '__main__':
    main()