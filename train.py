import numpy as np
import cv2
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    plt.imshow(X_train[0], cmap="gray")
    plt.show()
    print (y_train[0])
    ## Checking out the shapes involved in dataset
    print ("Shape of X_train: {}".format(X_train.shape))
    print ("Shape of y_train: {}".format(y_train.shape))
    print ("Shape of X_test: {}".format(X_test.shape))
    print ("Shape of y_test: {}".format(y_test.shape))
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    print ("Shape of X_train: {}".format(X_train.shape))
    print ("Shape of y_train: {}".format(y_train.shape))
    print ("Shape of X_test: {}".format(X_test.shape))
    print ("Shape of y_test: {}".format(y_test.shape))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    ## Declare the model
    model = Sequential()

    ## Declare the layers
    layer_1 = Conv2D(32, kernel_size=5, activation='relu', input_shape=(28, 28, 1))
    layer_2 = MaxPooling2D((2, 2))
    layer_3 = Conv2D(64, kernel_size=5, activation='relu')
    layer_4 = Flatten()
    layer_5 = Dense(10, activation='softmax')

    ## Add the layers to the model
    model.add(layer_1)
    model.add(layer_2)
    model.add(layer_3)
    model.add(layer_4)
    model.add(layer_5)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
 
    example = X_train[1]
    prediction = model.predict(example.reshape(1, 28, 28, 1))
    ## First output
    print ("Prediction (Softmax) from the neural network:\n\n {}".format(prediction))
    ## Second output
    hard_maxed_prediction = np.zeros(prediction.shape)
    hard_maxed_prediction[0][np.argmax(prediction)] = 1
    print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
    ## Third output
    print ("\n\n--------- Prediction --------- \n\n")
    plt.imshow(example.reshape(28, 28), cmap="gray")
    plt.show()
    print("\n\nFinal Output: {}".format(np.argmax(prediction)))
