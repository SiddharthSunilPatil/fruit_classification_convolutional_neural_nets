import os
import numpy as np
import keras
from src.logger import logging
from src.exception import CustomException

loaded_model=keras.saving.load_model('model\saved_model.keras')

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predictdata(self,image):
        try:
            logging.info('Entered the predict pipeline')
            predictions=loaded_model.predict(image)
            
            return predictions
        
        except Exception as e:
            raise CustomException(e,sys)
        