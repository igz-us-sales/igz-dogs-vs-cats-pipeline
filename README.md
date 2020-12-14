# Dogs vs Cats Pipeline with Kubeflow, MLRun, Nuclio, PyTorch, and S3
## Quick Links
1. [Getting Started](#Getting-Started)
2. [Inputs and Outputs](#Inputs-and-Outputs)
3. [Components Overview](#Components-Overview)
4. [Config Parameters Overview](#Config-Parameters-Overview)

## Getting Started
1. Upload Dogs vs Cats dataset to desired S3 location. Dataset can be found [here](https://github.com/igz-us-sales/dogs_vs_cats_data/tree/master/data).
2. Copy `config-default.yaml` as `config.yaml`.
3. Update `config.yaml` with paths and add AWS access keys. Config parameters will be explained below.
4. Run `DogsVsCatsPipeline.ipynb`, uncommenting any cells that are commented. Once the commented cells have been run once, they can be re-commented.

## Inputs and Outputs
### Inputs
- `*.csv`: File with paths to images in S3.

### Outputs
- `results.csv`: File with metrics/model info including model parameters, test/validation accuracy, train/validation/test loss, etc.
- `model_state_dict.pth`: File with trained parameters for PyTorch model. Exported state_dict vs whole model because it is best practices per PyTorch [documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html).
- `prep_model.py`: Snippet of code to load PyTorch model using `model_state_dict.pth`.

## Components Overview
### DownloadS3
   - Downloads image data from S3 to local filesystem using image paths from input  `.csv`.
   - Appends local file paths to  `.csv` and logs to MLRun DB for later consumption.
   - Option to bypass S3 download (given that input `.csv` has local file paths already) for the sake of limiting transfers.
   
### PrepData
   - Loads images from local filesystem.
   - Performs image transformations (resize, normalize, tensor conversion).
   - Splits data into train/validation/test sets.
   - Creates PyTorch DataLoaders for train/validation/test sets with given batch size.
   - Logs DataLoaders to MLRun DB for later consumption.
   
### TrainModel
   - Loads train/validation data from DataLoaders.
   - Initializes ResNet50 model for transfer learning on on Dogs vs Cats dataset.
   - Trains model using automated hyper-parameter tuning via MLRun and automatically selects best model based on validation accuracy.
   - Calculates training/validation loss and validation accuracy during training.
   - Logs results to MLRun DB.
   - Logs model with additional info attached (labels, metrics, model parameters) to MLRun DB.
   
### EvalModel
   - Loads test data from DataLoaders.
   -  Initializes ResNet50 model and loads trained parameters.
   - Calculates test loss/accuracy.
   - Logs metrics to MLRun DB.
   - Attaches testing metrics to model in MLRunDB.
   
### DeployModel
   - Serverless model endpoint via Nuclio
   - Initializes PyTorch model and loads state dict from previous pipeline components.
   - Takes base64 encoded image and performs prediction using model.
   - Sends back base64 encoded image with prediction.
   
### UploadS3
- Creates/uploads  `results.csv` with metrics/model info from MLRun DB.
- Creates/uploads `model_state_dict.pth` with trained model parameters from MLRun DB.
- Creates/uploads `prep_model.py` with code snippet to load PyTorch model using `model_state_dict.pth`.

## Config Parameters Overview
### Project
|Parameter|Type| Description|
|--|--|--|
| `name` | string | Name of MLRun project.  |
| `container` | string | Name of V3IO container where data will be downloaded/accessed/uploaded.  |
| `debug_logs` | string | Flag to print debug messages in components.  |

### Docker
|Parameter|Type| Description|
|--|--|--|
| `s3_image` | string | Name of custom Docker image to be build within platform.  |

### Data
|Parameter|Type| Description|
|--|--|--|
| `remote_download_path` | string | Path on local filesystem that S3 data will be downloaded to. Used by downstream components to read images.  |
| `mount_download_path` | string | Path on container filesystem that `remote_download_path` will mounted to. Allows containers to access data from local filesystem. |
| `download_data` | bool | Flag to bypass S3 download (given that input `.csv` has local file paths already) for the sake of limiting transfers. |
| `batch_size` | int | Size of batches created by DataLoader to load into model. |
| `img_dimensions` | int | Size to resize images to during image transformation. Model needs a square image (`img_dimensions` x `img_dimensions`). |
| `train_pct` | float | Percentage of data to be allocated to training set. |
| `val_pct` | float | Percentage of data to be allocated to validation set. |
| `test_pct` | float | Percentage of data to be allocated to testing set. |

### AWS
|Parameter|Type| Description|
|--|--|--|
| `bucket_name` | string | Name of S3 bucket where data is stored and results will be written.  |
| `aws_access_key_id` | string | AWS_ACCESS_KEY_ID for authentication.  |
| `aws_secret_access_key` | string | AWS_SECRET_ACCESS_KEY for authentication.  |
| `aws_default_region` | string | AWS region where bucket is located.  |
| `results_upload_path` | string | Path in S3 bucket where results files will be uploaded.  |

### Train
|Parameter|Type| Description|
|--|--|--|
| `epochs` | int | Number of epochs to train model.  |
| `lr` | list(float) | List of learning rates for hyper-parameter tuning. Used in training optimizer.  |
| `layer_size` | list(int) | List of layer sizes for hyper-parameter tuning. Adjusts model linear layer architecture.  |
| `device` | string | Device to train model and load images on (`cpu` or `cuda`).  |
| `hyper_param_runs` | int | Number of parameter permutations to run for hyper-parameter tuning.  |