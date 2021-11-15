from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization


#fitting the dataset
CNN_trainX = train_data.reshape(9748009, 120, 1)
CNN_trainY = to_categorical(train_labels, num_classes=138)
CNN_trainX.shape, CNN_trainY.shape


#Defining CNN
import tensorflow as tf

n_timesteps, n_features, n_outputs = CNN_trainX.shape[1], CNN_trainX.shape[2], CNN_trainY.shape[1]

model = Sequential()

model.add(Conv1D(filters= 16, kernel_size= 2, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(BatchNormalization())

model.add(Conv1D(filters= 64, kernel_size= 3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(BatchNormalization())


model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters= 32, kernel_size= 2, activation='relu',  kernel_regularizer= tf.keras.regularizers.l2(l=0.01)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters= 64, kernel_size= 3, activation='relu',  kernel_regularizer= tf.keras.regularizers.l2(l=0.01)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters= 128, kernel_size= 3, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
print(model.summary())

#Training the model
fit = model.fit(CNN_trainX[1:120000], CNN_trainY[1:120000], batch_size = 32,  verbose=True , epochs= 15)

#Accuracy
test = np.load("C:/bio/8th Sem/IDC 410 ML/Frame level Speech Recog/DATA/test_data_v120.npy", allow_pickle=True,encoding="latin1")
CNN_test = test.reshape(4620355, 120, 1)
y_pred = model.predict_classes(CNN_test)

#writing on a csv file for Kaggle submission
with open('C:/bio/8th Sem/IDC 410 ML/Frame level Speech Recog/DATA/Final_Submission6.csv', 'w') as w:
    w.write('id,label\n')
    for i in range(len(y_pred)):
            w.write(str(i)+','+str(y_pred[i])+'\n')
