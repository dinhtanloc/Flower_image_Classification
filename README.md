# Flower Classification Data Science Project

This project is for my Kaggle practice to enhance my skills.

## 1. Dataset

- [Kaggle Dataset for Image Classification](https://www.kaggle.com/datasets/marquis03/flower-classification)

### About the Dataset

The dataset consists of images of 14 types of flowers, with 13,618 training images and 98 validation images. The total dataset size is 202MB. It supports recognition of the following flower types: carnation, iris, bluebells, golden english, roses, fallen nephews, tulips, marigolds, dandelions, chrysanthemums, black-eyed daisies, water lilies, sunflowers, and daisies.

## 2. Project Structure Explanation

My project is structured into 7 folders:

- **config**: Contains paths used frequently in this project. You can set paths for storing experiment results or configuring model parameters such as experiments or models.
  
- **libs**: Includes a list of packages imported for this project.
  
- **utils**: Contains functions for preprocessing images using Keras and visualization functions used in this project.
  
- **models**: Includes CNN model architectures. You can download models from the Keras applications GitHub repository and import them for your project. Here, you can adjust parameters or load weights for each model in your experiments.
  
- **prj**: This main folder contains 4 strategies from EDA to training models, divided into four subfolders for easier tracking and management.

## 3. Experiment setup
This project focuses on implementing popular CNN models such as VGG16, ResNet50, etc. I use the TensorFlow framework with Python 3.9 to build and run these models. Due to limitations with my CPU, training each model can be time-consuming.
