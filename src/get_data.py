import pandas as pd
import os
import yaml
import argparse


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
        return config
    
def get_data(config_path):
    config = read_params(config_path)
    data_path = config["data_source"]["s3_source"]
    df = pd.read_csv(data_path,sep=",")
    print (df.head())
    return df

# extra code

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parseed_args = args.parse_args()
    data = get_data(config_path=parseed_args.config)
    