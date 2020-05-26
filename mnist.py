from keras.datasets import mnist
db=mnist.load_data("mymnist.dataset")
train, test=db
x_train,y_train=train
x_test,y_test=test
#import matplotlib.pyplot as plt
#plt.imshow(x_train[30987])
#y_train
#converting 2D into 1D
photo=x_train[0].reshape(28*28)
print("length of photo",len(photo))
x_train= x_train.reshape(-1,28*28)
x_test= x_test.reshape(-1,28*28)
from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
from keras.models import Sequential
model=Sequential()
from keras.layers import Dense
model.add(Dense(256,input_shape=(784,),activation='relu'))
model.add(Dense(230,activation='relu'))
model.add(Dense(150,activation='relu'))
model.add(Dense(64,activation='relu'))
#the output node is also treated as layer having softmac activation function
#because it is a multiclassification
model.add(Dense(10,activation='softmax'))
summary=model.summary()
print(summary)
from keras.optimizers import RMSprop
#this step will tell about optimizer, learning rate, loss, accuracy
model.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics=[('accuracy')])
s=model.fit(x_train,y_train,epochs=9)
h=model.predict(x_test)
print("accuracy:",s.history['accuracy'][8])
