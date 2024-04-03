import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from src.logger import logging
from src. exception import CustomException
from dataclasses import dataclass
from contextlib import redirect_stdout

#defining variables for model training
image_size=256
batch_size=32
channels=3
n_classes=16
epochs=50

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("model","saved_model.keras")
    model_summary_path=os.path.join("model","model_summary")
    model_performance_path=os.path.join("model","model_performance.png")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_path,val_path,test_path):
        try:
            logging.info("model trainer initiated")

            train_dataset=tf.data.Dataset.load(train_path)
            val_dataset=tf.data.Dataset.load(val_path)
            test_dataset=tf.data.Dataset.load(test_path)

            logging.info("Read the train, validation and test datasets")

            train_dataset=train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            val_dataset=val_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            test_dataset=test_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

            logging.info("optimization of the train, validation and test data completed")

            # resizing and rescaling the dataset

            Resizing=keras.layers.Resizing(image_size,image_size)
            Rescaling=keras.layers.Rescaling(1.0/255)

            #resize_and_rescale=tf.keras.Sequential([
                #layers.experimental.preprocessing.Resizing(image_size,image_size),
                #layers.experimental.preprocessing.Rescaling(1.0/255)
            #])

            logging.info("resizing and rescaling completed")

            Flipping=keras.layers.RandomFlip("horizontal_and_vertical")
            Rotating=keras.layers.RandomRotation(0.2)

            # data augmentation
            #data_augmentation=tf.keras.Sequential([
                #layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                #layers.experimental.preprocessing.RandomRotation(0.2)
            #])

            logging.info("data augmentation completed")

            #building the model
            input_shape=(batch_size,image_size,image_size,channels)
            
            model=keras.Sequential([
                Resizing, Rescaling, Flipping, Rotating,
                keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=input_shape),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Conv2D(16,(3,3),activation='relu'),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Conv2D(16,(3,3),activation='relu'),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Conv2D(32,(3,3),activation='relu'),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Conv2D(32,(3,3),activation='relu'),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Flatten(),
                keras.layers.Dense(64,activation='relu'),
                keras.layers.Dense(n_classes,activation='softmax')
            ])
            
            model.build(input_shape=input_shape)

            logging.info("model building completed")
            
            os.makedirs(os.path.dirname(self.model_trainer_config.model_summary_path),exist_ok=True)

            #cheking model summary
            with open(self.model_trainer_config.model_summary_path,'w') as f:
                with redirect_stdout(f):
                    model.summary()

            logging.info("saved model summary")

            #compiling the model
            model.compile(
                    optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy']
                )
            logging.info("model compilation completed")

            #fitting the model
            history=model.fit(
                train_dataset,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                validation_data=val_dataset
            )

            logging.info("fitting of model completed")

            #evaluating performance on test data
            scores=model.evaluate(test_dataset)
            print("Performance on test set:",scores)
            logging.info("evaluation of test dataset completed")

            # storing model history parameters in variables
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            train_loss = history.history['loss']
            val_loss = history.history['val_loss']

            # plotting model performance
            fig=plt.figure(figsize=(10,6))
            plt.subplot(1,2,1)
            plt.plot(range(epochs),train_acc,label='Training accuracy')
            plt.plot(range(epochs),val_acc,label='Validation accuracy')
            plt.legend(loc='lower right')
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.title('Training and Validation accuracy')

            plt.subplot(1,2,2)
            plt.plot(range(epochs),train_loss,label='Training loss')
            plt.plot(range(epochs),val_loss,label='validation loss')
            plt.legend(loc='upper right')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title('Training and Validation loss')

            fig.savefig(self.model_trainer_config.model_performance_path)
            logging.info("saved model performance")

            #saving the model
            model.save(self.model_trainer_config.trained_model_file_path)

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e,sys)