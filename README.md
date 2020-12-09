- Components
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





The flow should be:
- Upload a few images
- Post fs location to stream
- Nuclio function does some inferencing (with or withour gpus)

They key is to process images in batches. I think Tensorflow has a DataLoader that loads a batch of files in GPU for processing.  Then we can talk about how we can scale this with more shards, more workers, more replicas.

DataLoader:
- https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
- https://visualstudiomagazine.com/articles/2020/09/10/pytorch-dataloader.aspx
- https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
- https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data
- https://discuss.pytorch.org/t/how-to-load-images-with-images-names-using-pytorch/47408/4
- https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312/8

PyTorch:
- https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
- https://github.com/spmallick/learnopencv/blob/master/Image-classification-pre-trained-models/Image_Classification_using_pre_trained_models.ipynb

Dogs vs Cats:
- https://wtfleming.github.io/2020/04/12/pytorch-cats-vs-dogs-part-3/
- https://www.kaggle.com/jaeboklee/pytorch-cat-vs-dog