from typing import Set
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
   Dense,
   Embedding, 
   Dropout,
   Activation,
   Flatten
)
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint

from RAMPAGE.Classifier import Classifier
from RAMPAGE.Result import Result
from RAMPAGE.DataElement import DataElement
from common.resultCommon import ResultCommon
from common.commonData import CommonData


class BaselineExample(Classifier):
   """
   Baseline neural network classifier for domain name classification.
   
   A simple baseline model using embedding and dense layers to classify
   domains as DGA or legitimate.
   
   Attributes:
       model_name (str): Name identifier for the model.
       max_length (int): Maximum length of input sequences.
       save_file (str): Path to save the trained model.
   """

   def __init__(self) -> None:
       """Initialize the baseline classifier with a simple architecture."""
       self.model_name = "Baseline_example"
       self.commonData = CommonData()
       self.max_length = self.commonData.max_length
       self.save_file = f"./classifiers/models/{self.model_name}.keras"
       
       self._build_model()

   def _build_model(self) -> None:
       """Build and compile the baseline model architecture."""
       self.model = Sequential(name=self.model_name)
       
       # Simple embedding layer
       self.model.add(Embedding(
           input_dim=128,
           output_dim=128,
           input_length=self.max_length
       ))
       
       # Flatten for dense layer
       self.model.add(Flatten())
       
       # Output layer
       self.model.add(Dense(
           units=1,
           activation='sigmoid'
       ))
       
       # Model compilation with default Adam optimizer
       self.model.compile(
           loss='binary_crossentropy',
           optimizer='adam',
           metrics=self.commonData.metrics
       )

   def _prepare_data(self, data: Set[DataElement]) -> tuple[np.ndarray, np.ndarray]:
       """
       Convert domain data to numerical sequences.
       
       Args:
           data: Set of DataElement objects.
           
       Returns:
           tuple: (features, labels) as numpy arrays.
       """
       x_data = []
       y_data = []
       
       for element in data:
           # Convert domain characters to numerical values
           data_vec = [ord(char) - 33 for char in element.domain]  # 33 is '!'
           x_data.append(data_vec)
           y_data.append(1 if element.is_dga else 0)
       
       # Pad sequences and convert to numpy arrays
       x_data = pad_sequences(x_data, padding='post', maxlen=self.max_length)
       return np.array(x_data, dtype=float), np.array(y_data, dtype=float)

   def train(self, train_set: Set[DataElement], validation_set: Set[DataElement]) -> None:
       """
       Train the baseline model.
       
       Args:
           train_set: Training dataset.
           validation_set: Validation dataset.
       """
       # Prepare training and validation data
       x_train, y_train = self._prepare_data(train_set)
       x_val, y_val = self._prepare_data(validation_set)
       
       # Define model checkpoint for saving best model
       checkpoint = ModelCheckpoint(
           self.save_file,
           monitor='val_accuracy',
           verbose=self.commonData.verbose,
           save_best_only=True,
           save_weights_only=False,
           mode='auto'
       )
       
       # Train the model
       self.model.fit(
           x_train,
           y_train,
           epochs=self.commonData.epochs,
           verbose=self.commonData.verbose,
           validation_data=(x_val, y_val),
           batch_size=self.commonData.batch_size,
           callbacks=[checkpoint]
       )

   def test(self, test_set: Set[DataElement]) -> Result:
       """
       Test the trained model.
       
       Args:
           test_set: Test dataset.
           
       Returns:
           ResultCommon: Object containing evaluation metrics.
       """
       # Load the best model
       best_model = load_model(self.save_file)
       
       # Prepare test data
       x_test, y_test = self._prepare_data(test_set)
       
       # Evaluate model
       scores = best_model.evaluate(x_test, y_test, batch_size=10)
       
       # Return metrics
       return ResultCommon(
           accuracy=scores[1] * 100,
           precision=scores[2] * 100,
           recall=scores[3] * 100,
           fp=scores[4],
           fn=scores[5],
           tp=scores[6],
           tn=scores[7],
           auc=scores[8]
       )