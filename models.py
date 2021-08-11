import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

class CNN_Gender(tf.keras.Model):
    
    def __init__ (self):
        super(CNN_Gender, self).__init__()
        self.conv1 = Conv2D(32, (3,3), activation = 'relu', input_shape = (200,200,3))
        self.batchnorm1 = BatchNormalization(axis = -1)
        self.maxpool1 = MaxPooling2D()
        self.dropout1 = Dropout(0.2)
        self.conv2 = Conv2D(64, (3,3), activation = 'relu')
        self.batchnorm2 = BatchNormalization(axis = -1)
        self.maxpool2 = MaxPooling2D()
        self.dropout2 = Dropout(0.2)
        self.conv3 = Conv2D(128, (3,3), activation = 'relu')
        self.batchnorm3 = BatchNormalization(axis = -1)
        self.maxpool3 = MaxPooling2D()
        self.dropout3 = Dropout(0.2)
        self.flatten = Flatten()
        self.dense1 = Dense(32, activation = 'relu')
        self.dense2 = Dense(16,activation = 'relu')
        self.dense3 = Dense(1, activation = 'sigmoid')
        
    def call(self, inputs):
        
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x
    
class CNN_Age(tf.keras.Model):
    
    def __init__ (self):
        super(CNN_Age, self).__init__()
        self.conv1 = Conv2D(32, (3,3), activation = 'relu', input_shape = (200,200,3))
        self.batchnorm1 = BatchNormalization(axis = -1)
        self.maxpool1 = MaxPooling2D()
        self.dropout1 = Dropout(0.2)
        self.conv2 = Conv2D(64, (3,3), activation = 'relu')
        self.batchnorm2 = BatchNormalization(axis = -1)
        self.maxpool2 = MaxPooling2D()
        self.dropout2 = Dropout(0.2)
        self.conv3 = Conv2D(128, (3,3), activation = 'relu')
        self.batchnorm3 = BatchNormalization(axis = -1)
        self.maxpool3 = MaxPooling2D()
        self.dropout3 = Dropout(0.2)
        self.flatten = Flatten()
        self.dense1 = Dense(32, activation = 'relu')
        self.dense2 = Dense(16,activation = 'relu')
        self.dense3 = Dense(1, activation = 'linear')
        
    def call(self, inputs):
        
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x