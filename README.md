# Dogs vs Cats Pipeline with Kubeflow, MLRun, PyTorch, and S3
## Getting Started
1. Upload Dogs vs Cats dataset to desired S3 location. Dataset can be found [here](https://github.com/igz-us-sales/dogs_vs_cats_data/tree/master/data).
2. Copy `config-default.yaml` as `config.yaml`.
3. Update `config.yaml` with paths and add AWS access keys. Config parameters will be explained below.
4. Run `DogsVsCatsPipeline.ipynb`, uncommenting any cells that are commented. Once the commented cells have been run once, they can be re-commented.

## Inputs/Outputs
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
   
### UploadS3
- Creates/uploads  `results.csv` with metrics/model info from MLRun DB.
- Creates/uploads `model_state_dict.pth` with trained model parameters from MLRun DB.
- Creates/uploads `prep_model.py` with code snippet to load PyTorch model using `model_state_dict.pth`.

## Config Parameters Overview
|Parameter|Type| Description|
|--|--|--|
| asdf | asdf |asdf  |