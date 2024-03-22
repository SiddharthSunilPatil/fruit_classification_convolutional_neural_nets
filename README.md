## trial project


# Housing Price Advanced Regression
**Author:** Siddharth Patil.  

![image](https://github.com/SiddharthSunilPatil/house_price_advanced_regression/blob/main/static/house_price_pred_1.jpg
)

## Quicklinks
[notebook](https://github.com/SiddharthSunilPatil/house_price_advanced_regression/blob/main/Notebook/housingprice.ipynb)       
[AWS deployment link](http://trialproject-env.eba-mmr3didc.us-east-2.elasticbeanstalk.com/classify_image)  
[Dataset]( https://data.mendeley.com/datasets/bdd69gyhv8/1)

## Setup Instructions

**1. Cloning the repository**

1.1. Create a dirctory on your drive.  
1.2. Open anaconda prompt and navigate to the directory with the command "cd (type your directory path)".  
1.3. Launch VS code with command "code ."  
1.4. Open new terminal and use command "git clone https://github.com/SiddharthSunilPatil/house_price_advanced_regression.git
" to clone repository to existing directory.  
 
**2. Setting up the environment**  

2.1. Navigate to cloned repository with command "cd (type your repository relative path)".  
2.2. Create virtual environment with command "conda create -p venv python -y".  
2.3  Activate environment with command "conda activate venv/".  

**3. Installing dependencies**  

3.1. Use command "pip install -r requirements.txt" to install dependencies.  

**4. Downloading dataset**

4.1. Download the dataset from "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques".  
4.2. Create a new folder in your cloned repository and copy and paste the downloaded data to this folder.    

**5. Training the model**  

5.1. Copy the relative paths for train and test data from step 4.2.    
5.2. In the src/components/data_ingestion.py file, paste these relative paths in line 26 and 27 as shown below.  
            train_df=pd.read_csv('relative path for train dataset').  
            test_df=pd.read_csv('relative path for test dataset').  
5.3. Execute command "python src/components/data_ingestion.py".      
5.4  After completion of code execution, an artifacts folder will be created with 3 files viz "concat.csv", "model.pkl" and "preprocessor.pkl".  

**6. Deploying the model to local server with Flask**

6.1. Execute command "python application.py".  
6.2. The application will be served on localhost and is ready to use.  

## Business Problem

Housing price prediction is becoming an increasingly valuable tool for various stakeholders such as investors, homebuyers and government institutions, facilitating informed decision-making, risk management, and economic planning. A large number of factors influence house price negotiations and not just the number of bedrooms or total area. The aim of the project is to develop an algorithm for estimating the correct price of a house based on 80+ variables. 

## Data

The dataset is based on information collected from actual sales of 2919 houses in the Ames city in Iowa state. The dataset was obtained from kaggle.

#### Datasource link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

### The dependent or target variables is:
1. **SalePrice** - The property's sale price in dollars. This is the target variable that you're trying to predict.

### The independent variables or features are as below
2. **MSSubClass:** The building class
3. **MSZoning:** The general zoning classification
4. **LotFrontage:** Linear feet of street connected to property
5. **LotArea:** Lot size in square feet
6. **Street:** Type of road access
7. **Alley:** Type of alley access
8. **LotShape:** General shape of property
9. **LandContour:** Flatness of the property
10. **Utilities:** Type of utilities available
11. **LotConfig:** Lot configuration
12. **LandSlope:** Slope of property
13. **Neighborhood:** Physical locations within Ames city limits
14. **Condition1:** Proximity to main road or railroad
14. **Condition2:** Proximity to main road or railroad (if a second is present)
15. **BldgType:** Type of dwelling
16. **HouseStyle:** Style of dwelling
17. **OverallQual:** Overall material and finish quality
18. **OverallCond:** Overall condition rating
20. **YearBuilt:** Original construction date
21. **YearRemodAdd:** Remodel date
22. **RoofStyle:** Type of roof
23. **RoofMatl:** Roof material
24. **Exterior1st:** Exterior covering on house
25. **Exterior2nd:** Exterior covering on house (if more than one material)
26. **MasVnrType:** Masonry veneer type
27. **MasVnrArea:** Masonry veneer area in square feet
28. **ExterQual:** Exterior material quality
29. **ExterCond:** Present condition of the material on the exterior
30. **Foundation:** Type of foundation
31. **BsmtQual:** Height of the basement
32. **BsmtCond:** General condition of the basement
33. **BsmtExposure:** Walkout or garden level basement walls
34. **BsmtFinType1:** Quality of basement finished area
35. **BsmtFinSF1:** Type 1 finished square feet
36. **BsmtFinType2:** Quality of second finished area (if present)
37. **BsmtFinSF2:** Type 2 finished square feet
38. **BsmtUnfSF:** Unfinished square feet of basement area
39. **TotalBsmtSF:** Total square feet of basement area
40.  **Heating:** Type of heating
41. **HeatingQC:** Heating quality and condition
42. **CentralAir:** Central air conditioning
43. **Electrical:** Electrical system
44. **1stFlrSF:** First Floor square feet
45. **2ndFlrSF:** Second floor square feet
46. **LowQualFinSF:** Low quality finished square feet (all floors)
47. **GrLivArea:** Above grade (ground) living area square feet
48. **BsmtFullBath:** Basement full bathrooms
49. **BsmtHalfBath:** Basement half bathrooms
50. **FullBath:** Full bathrooms above grade
51. **HalfBath:** Half baths above grade
52. **Bedroom:** Number of bedrooms above basement level
53. **Kitchen:** Number of kitchens
54. **KitchenQual:** Kitchen quality
55. **TotRmsAbvGrd:** Total rooms above grade (does not include bathrooms)
56. **Functional:** Home functionality rating
57. **Fireplaces:** Number of fireplaces
58. **FireplaceQu:** Fireplace quality
59. **GarageType:** Garage location
60. **GarageYrBlt:** Year garage was built
61. **GarageFinish:** Interior finish of the garage
62. **GarageCars:** Size of garage in car capacity
63. **GarageArea:** Size of garage in square feet
64. **GarageQual:** Garage quality
65. **GarageCond:** Garage condition
66. **PavedDrive:** Paved driveway
67. **WoodDeckSF:** Wood deck area in square feet
68. **OpenPorchSF:** Open porch area in square feet
69. **EnclosedPorch:** Enclosed porch area in square feet
70. **3SsnPorch:** Three season porch area in square feet
71. **ScreenPorch:** Screen porch area in square feet
72. **PoolArea:** Pool area in square feet
73. **PoolQC:** Pool quality
74. **Fence:** Fence quality
75. **MiscFeature:** Miscellaneous feature not covered in other categories
76. **MiscVal:** Value of miscellaneous feature
77. **MoSold:** Month Sold
78. **YrSold:** Year Sold
79. **SaleType:** Type of sale
80. **SaleCondition:** Condition of sale
 
## Project Approach

**1. Data Ingestion:** In this phase, both test and train dataset are read from csv file. Since both train set and test set contain missing values, data is first concatenated and prepared for analysis.

**2. Data Transformation:** The concatenated data is then passed through a column transformer pipeline. Based on the variable the transformer performs various preprocessing treatments such as simple imputing, nominal encoding, ordinal encoding and stadard scaling. The transformed data is then passed to the model trainer.

**3. Model Trainer:** Various models are tested to find out the best perfroming model based on Rott Mean Squred Error[RMSE] score. The best model found was Gradient Boosting Regressor. The model is then improved by hyperparameter tuning.

**4. Prediciton Pipeline:** This pipleline converts input data into a dataframe and loads pickle files for data transformation and model training and predicts final results.

**5. Deployment:** The project is deployed on amazon elastic beanstalk as a flask application to predict house prices.








