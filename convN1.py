#Lucas Kermessi - ES98872 - INF420 PER 3
#Versão do Googlecolab com o resultado do código: https://colab.research.google.com/drive/1jlVx-eyX-09p-J-xmkgkxQx6P1YbIw9S?usp=sharing
#Acessar com o email da ufv

from time import time
import matplotlib.pyplot as plt
import numpy
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras import backend as K
K.set_image_data_format('channels_first')


# Fixando a aleatoriedade
seed = 7
numpy.random.seed(seed)

# Carregando os dados
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#print("Xtrain[0] ",X_train[0])
#print("Ytrain[0] ",y_train[0])


# Normalizando as entradas para 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# Codificando a saida para one hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#print("Xtrain[0] ",X_train[0])
#print("Ytrain[0] ",y_train[0])


# Criando o modelo
model = Sequential()
model.add(Conv2D(filters=32, kernel_size= (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size= (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size= (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size= (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size= (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size= (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compilando o modelo
epocas = 20
lrate = 0.001
sgd = SGD(learning_rate=lrate)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy' ]) 
print(model.summary())


# Treinando o modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epocas, batch_size=32)

# Avaliacao final do modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Sumariza para a precisão
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Sumariza para a mostrar a perda
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('modelT1')

print("\nFim")