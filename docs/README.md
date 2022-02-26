
# docs

## VISSL

If you want to know about VISSL, visit their [github](https://github.com/facebookresearch/vissl) or [documentation](https://vissl.readthedocs.io/en/main/).

## Usage

### Dataset

In my enviornment all images in Danbooru2020 is downloaded and resized so that the long side has 512 pixels.
They are placed inside `/mnt/D/Data/danbooru/resized/`, and mounted to docker images with `volumes`.
File paths are split into two, each for training and validation, and saved to a file with `numpy.save()`. See [`tools/splitdata.py`](../tools/splitdata.py).
These files are placed inside [`filelists/`](../filelists/), and the dataset is registered inside [`tools/run_distributed_engines.py`](../tools/run_distributed_engines.py)
Then the config file is edited so that it uses the registered dataset.

See [Using Custom Datasets](https://vissl.readthedocs.io/en/main/extend_modules/custom_datasets.html) in VISSL's documentation.

### Train

- First

    To give permission to read/write files/folders that were made inside the docker container to the local user.

    ```console
    export USERID=${UID}
    ```

- Build image

    ```console
    docker-compose build
    ```

    The build installs
    - [VISSL](https://github.com/facebookresearch/vissl) (main branch)
    - [ClassyVision](https://github.com/facebookresearch/ClassyVision)
    - [fairscale](https://github.com/facebookresearch/fairscale)
    - [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)

- Train models

    1. Add config file to [`configs/config/`](../configs/config/) like [`simclr_resnet.yaml`](../configs/config/simclr_resnet.yaml)

    2. Command to start training (sample):

    ```console
    docker-compose run vissl python -m tools.run_distributed_engines \
        config=simclr_resnet \
        config.CHECKPOINT.DIR="./checkpoint/simclr_resnet50" \
        config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=128 \
        config.MODEL.AMP_PARAMS.USE_AMP=true
    ```

- Train models not implemented in VISSL

    See [Add new Models](https://vissl.readthedocs.io/en/main/extend_modules/models.html) for how to implement new models that can be trained with VISSL.
    If you place the python file with the implemented model in [`models/visslmodels/`](../models/visslmodels/), it can be specified inside your configuration file.

    In this repository, I place the python file with the implementation of the model which is completely independent from VISSL inside [`models/torchmodels/`](../models//torchmodels/). Then, wrap the model to fit the VISSL requirements inside [`models/visslmodels/`](../models/visslmodels/). This is helpful when you want to use the model in an environment without VISSL installed.

### Pretrained weights

```console
docker-compose run vissl python -m tools.download_pretrained \
    --name simclr_resnet50 --out-folder ./weights
```

Use `--help` option for name choices.
