# Fruit classification convolutional neural network
**Author:** Siddharth Patil.  

![image](https://github.com/SiddharthSunilPatil/fruit_classification_convolutional_neural_nets/blob/main/static/fruit_img_2.png)

## Quicklinks
[Jupyter notebook](https://github.com/SiddharthSunilPatil/fruit_classification_convolutional_neural_nets/blob/main/notebook/Fresh_Rotten_Fruit_Classification.ipynb)       
[AWS deployment link](http://trialproject-env.eba-mmr3didc.us-east-2.elasticbeanstalk.com/classify_image)  
[Dataset]( https://data.mendeley.com/datasets/bdd69gyhv8/1)

## Setup Instructions

**1. Cloning the repository**

1.1. Create a dirctory on your drive.  
1.2. Open anaconda prompt and navigate to the directory with the command "cd (type your directory path)".  
1.3. Launch VS code with command "code ."  
1.4. Open new terminal and use command "git clone https://github.com/SiddharthSunilPatil/fruit_classification_convolutional_neural_nets.git
" to clone repository to existing directory.  
 
**2. Setting up the environment**  

2.1. Navigate to cloned repository with command "cd (type your repository relative path)".  
2.2. Create virtual environment with command "conda create -p venv python -y".  
2.3  Activate environment with command "conda activate venv/".  

**3. Installing dependencies**  

3.1. Use command "pip install -r requirements.txt" to install dependencies.  

**4. Downloading dataset**

4.1. Download the dataset from "https://data.mendeley.com/datasets/bdd69gyhv8/1".  
4.2. Create a new folder in your cloned repository and rename it to "dataset". Copy and paste the 16 subfolders after unzipping the  downloaded data to this folder.    

**5. Training the model**  

5.1. Execute command "python src/components/data_ingestion.py".      
5.2  After completion of code execution, an artifacts folder will be created containing 3 files viz train_dataset, test_dataset and val_dataset. A model folder containing the trained model, model performance and model summary will also be created 

**6. Deploying the model to local server with Flask**

6.1. Execute command "python application.py".  
6.2. The application will be served on localhost and is ready to use.  

## Business Problem

In the fruit industry, ensuring the freshness and quality of produce is paramount to both economic viability and consumer health perceptions. However, the prevalent manual classification methods for distinguishing between fresh and rotten fruits are inefficient and prone to errors, leading to significant financial losses for farmers, sellers, and processing industries. Additionally, consumer trust and sales are adversely affected by the presence of spoiled fruits in the market.

The primary objective is to develop robust and accurate computer vision algorithms capable of reliably distinguishing between fresh and rotten fruits across various types. By implementing these algorithms, stakeholders in the fruit industry can streamline quality control processes, minimize financial losses due to spoilage, and enhance consumer confidence by ensuring the delivery of fresh and safe produce.

## Data

The dataset was sourced from Medeley data. 

#### Dataset link: https://data.mendeley.com/datasets/bdd69gyhv8/1

**citation:**Sultana, Nusrat; Jahan, Musfika; Uddin, Mohammad Shorif (2022), “Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality”, Mendeley Data, V1, doi: 10.17632/bdd69gyhv8.1

The dataset is a collection of 3200 images from 8 different fruit categories viz apple, banana, grape, guava, jujube, orange, pomegranate and strawberry and belonging to 2 classes viz rotten and fresh.

 
## Project Approach

**1. Data Ingestion:** In this phase, the dataset is read using the keras preprocessing library and split into training dataset, test dataset and validation dataset

**2. Data transformation** Using the keras layers library, a pipeline is defined for resizing and rescaling the dataset to 256 X 256 pixels. A second pipeline for data augmentation for creating more variation is defined by using random rotation and random flip. These pipelines are called in the model compiler.

**3. Model compliation** A CNN model is compiled using the keras sequential library consisting of multiple layers. The starting layer is the data transformation pipeling described in step 2 followed by 5 sets of conv2D and maxpooling layers and with the flatten layer at the end. 

**3. Model Trainer:** A model is compiled comprising multiple layers 

**4. Prediciton Pipeline:** This pipleline converts input data into a dataframe and loads pickle files for data transformation and model training and predicts final results.

**5. Deployment:** The project is deployed on amazon elastic beanstalk as a flask application to predict house prices.








