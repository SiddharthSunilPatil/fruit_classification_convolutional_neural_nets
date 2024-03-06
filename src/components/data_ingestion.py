import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import tensorflow as tf
import keras



from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


#defining variables for data ingestion
image_size=256
batch_size=32

#defining function for splitting the dataset into train, validation and test set
def split_data(data,train_split=0.8,val_split=0.1,shuffle_size=10000,shuffle=True):
        data_size=len(data)
        if shuffle:
            data=data.shuffle(shuffle_size,seed=15)
        
        train_size=int(data_size*train_split)
        train_dataset=data.take(train_size)
        
        val_size=int(data_size*val_split)
        val_dataset=data.skip(train_size).take(val_size)
        
        test_size=data_size-(train_size+val_size)
        test_dataset=data.skip(train_size).skip(val_size)
        
        return train_dataset,val_dataset,test_dataset

#defining data ingestion configuration
@dataclass
class DataIngestionConfig:
     train_data_path=os.path.join("artifacts","train_dataset")
     val_data_path=os.path.join("artifacts","val_dataset")
     test_data_path=os.path.join("artifacts","test_dataset")

#ingesting the dataset
class DataIngestion: 
    def __init__(self):
         self.ingestion_config=DataIngestionConfig()
         

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")    
        try:
            dataset=keras.preprocessing.image_dataset_from_directory(
                "dataset",
                shuffle=True,
                image_size=(image_size,image_size),
                batch_size=batch_size
            )
            logging.info("Read the dataset as a tensor")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            class_names=dataset.class_names
            
            logging.info("Splitting data into train set, validation set and test set initiated")
            train_dataset,val_dataset,test_dataset=split_data(dataset)

            tf.data.Dataset.save(train_dataset,self.ingestion_config.train_data_path)
            tf.data.Dataset.save(val_dataset,self.ingestion_config.val_data_path)
            tf.data.Dataset.save(test_dataset,self.ingestion_config.test_data_path)

            logging.info("Ingestion and splitting of data completed")

            return(
                 self.ingestion_config.train_data_path,
                 self.ingestion_config.val_data_path,
                 self.ingestion_config.test_data_path,
                 class_names
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
     print("Entered method")
     obj=DataIngestion()
     train_path,val_path,test_path,class_names=obj.initiate_data_ingestion()

     model_trainer=ModelTrainer()
     model_trainer.initiate_model_trainer(train_path,val_path,test_path)