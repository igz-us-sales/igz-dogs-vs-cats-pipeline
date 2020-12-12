
import os
from kfp import dsl
from mlrun import mount_v3io, NewTask
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# image = f"docker-registry.{os.getenv('IGZ_NAMESPACE_DOMAIN')}:80/inference-benchmarking-demo"
# image = "mlrun/mlrun"
funcs = {}

# Configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    
    for fn in functions.values():
        # Mount V3IO filesystem
        fn.apply(mount_v3io())
        fn.apply(mount_v3io(name="csv",
                            remote=config["csv"]["s3_images_csv_remote_path"],
                            mount_path=config["csv"]["s3_images_csv_mount_path"]))
        fn.apply(mount_v3io(name="data",
                            remote=config["data"]["remote_download_path"],
                            mount_path=config["data"]["mount_download_path"]))
    
    # Set env var configuation for S3 functions
    s3_functions = ['download-s3', 'upload-s3']
    for func in s3_functions:
        functions[func].set_env('AWS_ACCESS_KEY_ID', config['aws']['aws_access_key_id'])
        functions[func].set_env('AWS_SECRET_ACCESS_KEY', config['aws']['aws_secret_access_key'])
        functions[func].set_env('AWS_DEFAULT_REGION', config['aws']['aws_default_region'])
       
    # Set GPU reources for model training
    #functions['train-model'].with_limits(gpus="1", gpu_type='nvidia.com/gpu')

    # Set resources for model deployment
    functions["deploy-model"].spec.base_spec['spec']['build']['baseImage'] = "mlrun/ml-models-gpu"
    functions["deploy-model"].spec.base_spec['spec']['loggerSinks'] = [{'level': 'info'}]
    functions["deploy-model"].spec.min_replicas = 1
    functions["deploy-model"].spec.max_replicas = 1

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name="Dogs vs Cats Pipeline",
    description="Kubeflow Pipeline Demo with PyTorch on Dogs vs Cats Dataset"
)
def kfpipeline(bucket_name:str = config['aws']['bucket_name'],
               s3_images_csv:str = f'{config["csv"]["s3_images_csv_mount_path"]}/{config["csv"]["s3_images_csv"]}',
               data_download_path:str = config['data']['mount_download_path'],
               results_upload_path:str = config['aws']['results_upload_path'],
               download_data:bool= config['data']['download_data'],
               batch_size:int = config['data']['batch_size'],
               img_dimensions:int = config['data']['img_dimensions'],
               train_pct:float = config['data']['train_pct'],
               val_pct:float = config['data']['val_pct'],
               test_pct:float = config['data']['test_pct'],
               epochs:int = config['train']['epochs'],
               lr:list = config['train']['lr'],
               layer_size:list = config['train']['layer_size'],
               hyper_param_runs:int = config['train']['hyper_param_runs'],
               device:str = config['train']['device'],
               debug_logs:bool= config['project']['debug_logs']):    
    
    # Download Data from S3
    inputs = {"bucket_name" : bucket_name,
              "s3_images_csv" : s3_images_csv,
              "data_download_path" : data_download_path,
              "download_data" : download_data}
    download_s3 = funcs['download-s3'].as_step(handler="handler",
                                               inputs=inputs,
                                               outputs=["s3_image_csv_local",
                                                        "data_download_path"],
                                               verbose=debug_logs)
    
    # Prep Data
    inputs = {"data_download_path" : download_s3.outputs['data_download_path'],
              "batch_size" : batch_size,
              "img_dimensions" : img_dimensions,
              "train_pct" : train_pct,
              "val_pct" : val_pct,
              "test_pct" : test_pct}
    prep_data = funcs['prep-data'].as_step(handler="handler",
                                           inputs=inputs,
                                           outputs=["train_data_loader",
                                                    "validation_data_loader",
                                                    "test_data_loader"],
                                           verbose=debug_logs)
    
    # Train Model
    inputs = {"train_data_loader" : prep_data.outputs["train_data_loader"],
              "validation_data_loader" : prep_data.outputs["train_data_loader"],
              "epochs" : epochs,
              "batch_size": batch_size,
              "device" : device}
    hyper_params = {'lr': lr,
                    "layer_size" : layer_size,
                    "MAX_EVALS": hyper_param_runs}
    train_model = funcs['train-model'].as_step(handler="handler",
                                               inputs=inputs,
                                               hyperparams=hyper_params,
                                               runspec=NewTask(tuning_strategy="random"),
                                               selector="max.validation_accuracy",
                                               outputs=["model"],
                                               verbose=debug_logs)
    
    # Evaluate Model
    inputs = {"test_data_loader" : prep_data.outputs["test_data_loader"],
              "model" : train_model.outputs["model"],
              "device" : device}
    eval_model = funcs['eval-model'].as_step(handler="handler",
                                             inputs=inputs,
                                             verbose=debug_logs)
    
    # Deploy Model
    env = {"model_url" : train_model.outputs["model"],
           "device" : device,
           "img_dimensions" : img_dimensions}
    deploy = funcs['deploy-model'].deploy_step(env=env)
    
    # Upload Model/Metrics to S3
    inputs = {"model" : train_model.outputs["model"],
              "bucket_name" : bucket_name,
              "results_upload_path" : results_upload_path}
    upload_s3 = funcs['upload-s3'].as_step(handler="handler",
                                           inputs=inputs,
                                           verbose=debug_logs)
    upload_s3.after(eval_model)
