# Dogs vs Cats Pipeline with Kubeflow, MLRun, PyTorch, and Nuclio
## Getting Started
1. Upload Dogs vs Cats dataset to desired S3 location. Dataset can be found [here](https://github.com/igz-us-sales/dogs_vs_cats_data/tree/master/data).
2. Copy `config-default.yaml` as `config.yaml`.
3. Update `config.yaml` with paths and add AWS access keys. Config parameters will be explained below.
4. Run `DogsVsCatsPipeline.ipynb`, uncommenting any cells that are commented. Once the commented cells have been run once, they can be re-commented.

## Components
- CSV with S3 paths and classifications
- DownloadS3
    - Download data from CSV
    - Append local paths to CSV
- PrepData
    - Load images from local paths via CSV
    - Perform image transformations and splits 
    - Output data loaders (train, validation, test)
- TrainModel
    - Input CSV
    - Output model to V3IO
- EvalMmodel
    - Input CS
- DeployModel
    - Model endpoint
- UploadS3
    - Upload model
    - Upload model metrics

### Config Parameters
