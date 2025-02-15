# VCS Project
## Enrica Bongiovanni, Laura Prendin, Arianna Stropeni

### Git Structure
The project includes the following files:

- [Utils](utils.py): it defines the functions required to create the datasets and the metrics used
- [Network layers](network_layers.py): it defines the functions for the custom layers used in the networks
- [Base U-Net](base_unet.ipynb), [Attention U-Net](attention_unet.ipynb), [UC-TransNet](uc_transnet.ipynb): they contain the architecture definition and the training on the three datasets of the different models
- [Evaluation](evaluate.ipynb): it includes the evaluation of the best models on the test set and the visualization of the results
- [Best Models](Best_models): the folder includes files .h5 that store the best models for each dataset and each architecture

### Datasets
To run the code, download the datasets from the following links, unzip them and store them in a "Datasets" folder in the repository:
- [Breast cancer dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- [Skin cancer dataset](https://challenge.isic-archive.com/data/#2017): download training, validation and test data and the associated ground truths only for task 1
- [Brain cancer dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation/data): rename the unzipped file "Brain_tumor_dataset"
