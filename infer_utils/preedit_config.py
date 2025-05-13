import os
import yaml

def conf_editor(config_path):

    class IndentDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentDumper, self).increase_indent(flow, False)


    def tuple_constructor(loader, node):
        # Load the sequence of values from the YAML node
        values = loader.construct_sequence(node)
        # Return a tuple constructed from the sequence
        return tuple(values)

    # Register the constructor with PyYAML  
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
tuple_constructor)



    def conf_edit(config_path):
        with open(config_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        # handle cases where 'use_amp' is missing from config:
        if 'use_amp' not in data.keys():
          data['training']['use_amp'] = True

        data['inference']['num_overlap'] = 2

        if data['inference']['batch_size'] == 1:
          data['inference']['batch_size'] = 2

        print("Using custom overlap and chunk_size values:")
        print(f"batch_size = {data['inference']['batch_size']}")


        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)
