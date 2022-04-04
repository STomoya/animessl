
# animessl

Training vision models with vissl + illustrated images.

## Models

|model|SSL|config file|
|-|-|-|
|ResNet50|SimCLR|[`./configs/config/simclr_resnet.yaml`](./configs/config/simclr_resnet.yaml)|
|ResNet50|SwAV|[`./configs/config/swav_resnet.yaml`](./configs/config/swav_resnet.yaml)

## Dataset

- Danbooru2021

    ```
    @misc{danbooru2021,
        author = {Anonymous and Danbooru community and Gwern Branwen},
        title = {Danbooru2021: A Large-Scale Crowdsourced and Tagged Anime Illustration Dataset},
        howpublished = {\url{https://www.gwern.net/Danbooru2021}},
        url = {https://www.gwern.net/Danbooru2021},
        type = {dataset},
        year = {2022},
        month = {January},
        timestamp = {2022-01-21},
        note = {Accessed: 2022-04-04} }
    ```

## Author

[STomoya](https://github.com/STomoya)
