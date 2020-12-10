
import os
from kfp import dsl
from mlrun import mount_v3io
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# image = f"docker-registry.{os.getenv('IGZ_NAMESPACE_DOMAIN')}:80/inference-benchmarking-demo"
# image = "mlrun/mlrun"
funcs = {}

# Configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    
    # Mount V3IO filesystem
    for fn in functions.values():
        fn.apply(mount_v3io())
        fn.apply(mount_v3io(name="csv",
                            remote=config["csv"]["s3_images_csv_remote_path"],
                            mount_path=config["csv"]["s3_images_csv_mount_path"]))
        fn.apply(mount_v3io(name="data",
                            remote=config["data"]["remote_download_path"],
                            mount_path=config["data"]["mount_download_path"]))
    
    # Set env var configuation for S3 functions
    s3_functions = ['download-s3']
    for func in s3_functions:
        functions[func].set_env('AWS_ACCESS_KEY_ID', config['aws']['aws_access_key_id'])
        functions[func].set_env('AWS_SECRET_ACCESS_KEY', config['aws']['aws_secret_access_key'])
        functions[func].set_env('AWS_DEFAULT_REGION', config['aws']['aws_default_region'])

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name="Dogs vs Cats Pipeline",
    description="Kubeflow Pipeline Demo with PyTorch on Dogs vs Cats Dataset"
)
def kfpipeline(bucket_name:str = config['aws']['bucket_name'],
               s3_images_csv:str = f'{config["csv"]["s3_images_csv_mount_path"]}/{config["csv"]["s3_images_csv"]}',
               data_download_path:str = config['data']['mount_download_path'],
               download_data:bool=False,
               debug_logs:bool=True):    
    
    # Download data
    inputs = {"bucket_name" : bucket_name,
              "s3_images_csv" : s3_images_csv,
              "data_download_path" : data_download_path,
              "download_data" : download_data}
    download_s3 = funcs['download-s3'].as_step(handler="handler",
                                               inputs=inputs,
                                               outputs=["s3_image_csv_local"],
                                               verbose=debug_logs)
    
    # Prep Data
    inputs = {"local_images_csv" : download_s3.outputs['s3_image_csv_local']}
    prep_data = funcs['prep-data'].as_step(handler="handler",
                                           inputs=inputs,
                                           verbose=debug_logs)
