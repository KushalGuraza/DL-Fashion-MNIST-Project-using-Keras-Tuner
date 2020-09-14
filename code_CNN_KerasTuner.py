import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sns


tf.__version__

#Loading the dataset
data = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images, test_labels) = data.load_data() 

#to check the number of output classes
sns.countplot(train_labels)

#scaling down the Images
train_images = train_images/255.0
test_images = test_images/255.0

train_images[0]
train_images[0].shape


#reshaping 
train_images = train_images.reshape(len(train_images),28,28,1)
test_images = test_images.reshape(len(test_images),28,28,1) 



def build_model(hp):
     model = keras.Sequential([ 
            
        keras.layers.Conv2D(
                filters = hp.Int('conv_1_filter', min_value = 32, max_value = 128 , step = 16),
                kernel_size = hp.Choice('conv_1_kernel', values = [3,5]),
                activation= 'relu',
                input_shape = (28,28,1)        
               ),
        
        keras.layers.Conv2D(
                filters = hp.Int('conv_2_filter', min_value = 32, max_value = 64, step = 16),
                kernel_size = hp.Choice('conv_2_kernel' , values = [3,5]),
                activation='relu'
                ),
        
        keras.layers.Flatten(),
        
        keras.layers.Dense(
                units = hp.Int('dense_1_units', min_value= 32, max_value = 128, step = 16),
                activation = 'relu'
                ),
        
        keras.layers.Dense(10, activation = 'softmax'),
    ])
         
     model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2,1e-3])),
                  loss = 'sparse_categorical_crossentropy',
                  metrics =['accuracy'])

     return model
 
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

 
tuner = RandomSearch(build_model, objective ='val_accuracy', max_trials=5,
                     directory='model',
                     project_name= 'fashion_mnist')


tuner.search(train_images, train_labels, epochs = 3 , validation_split = 0.1)

#choosing the top model out of 5
model = tuner.get_best_models(num_models = 1)[0]


#Fitting the data from 4th epoch  
model.fit(train_images, train_labels, epochs = 10, initial_epoch = 3, validation_split = 0.1)

#model saving
model.save('FASHION_MNIST.h5')

#predicting for the test data
y = model.predict_classes(test_images)

#calculating accuracy score of the model
from sklearn.metrics import accuracy_score, confusion_matrix
print( "                                                 ")
print( "*************************************************")
print( "                                                 ") 
print('          ACCURACY      ' + str(accuracy_score(test_labels, y)*100) +'%')
print( "                                                 ")
print( "*************************************************")
print( "                                                 ")

#summary of the model 
model.summary()

confusion_matrix(test_labels, y)







