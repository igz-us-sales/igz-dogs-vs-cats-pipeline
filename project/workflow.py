
import os
from kfp import dsl
from mlrun import mount_v3io

image = f"docker-registry.{os.getenv('IGZ_NAMESPACE_DOMAIN')}:80/inference-benchmarking-demo"
funcs = {}

# Configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    
    for fn in functions.values():
        # Apply V3IO mount
        fn.apply(mount_v3io(name="model",
                            remote="users/nick/igz_repos/igz-inference-benchmark/models",
                            mount_path="/model"))
        fn.apply(mount_v3io(name="stream",
                            remote="bigdata/dogs_vs_cats/data/",
                            mount_path="/stream"))
            
        # Set resources for jobs
        if fn.to_dict()["kind"] == "job":
            fn.spec.build.image = image
            
        # Set resources for nuclio functions
        elif fn.to_dict()["kind"] == "remote":
            fn.spec.base_spec['spec']['build']['baseImage'] = image
            fn.spec.base_spec['spec']['loggerSinks'] = [{'level': 'info'}]
            fn.spec.min_replicas = 1
            fn.spec.max_replicas = 1
        
    # Apply V3IO trigger
    image_igestion_trigger_spec={
        'kind': 'v3ioStream',
        'url' : f"http://v3io-webapi:8081/bigdata/dogs_vs_cats/stream@processorgrp",
        "password": os.getenv('V3IO_ACCESS_KEY'),  
        'attributes': {"pollingIntervalMs": 500,
            "seekTo": "earliest",
            "readBatchSize": 100,
            "partitions": "0-100",                          
          }
    }
    functions['inference-benchmark'].add_trigger('image-proc', image_igestion_trigger_spec)
#     functions['inference-benchmark'].with_limits(gpus="1", gpu_type='nvidia.com/gpu')

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name="Inference Benchmark",
    description="Benchmark Model Inferencing with CPUs vs GPUs"
)
def kfpipeline(model_path:str = '/model/dogs_vs_cats_resnet50.pth',
               stream_path:str = "dogs_vs_cats/stream",
               table_path:str = "dogs_vs_cats/table",
               batch_size:int = 32,
               num_batches:int = 8,
               device:str = "cpu"):

#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    inputs = {"stream_path" : stream_path}
    create_stream = funcs['create-stream'].as_step(handler="handler", inputs=inputs, outputs=["stream_url"])
    
    env = {"model_path" : model_path,
           "stream_path" : create_stream.outputs["stream_url"],
           "table_path" : table_path,
           "batch_size" : batch_size,
           "num_batches" : num_batches,
           "device" : device}
    
    # Inference benchmarking on GPU/CPU
    benchmark = funcs['inference-benchmark'].deploy_step(env=env)
