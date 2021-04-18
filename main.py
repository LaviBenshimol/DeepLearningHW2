
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2
import h5py
import time

#from preprocess_data import dataloader

from tqdm import tqdm
from math import ceil

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from skimage.transform import rotate, AffineTransform, warp, rescale
#from skimage.util import random_noise

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Flatten, Dense, Concatenate, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
from statistics import mean

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)



def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
    result = K.maximum(sum_square, K.epsilon())
    return result

class SiameseNetwork():

    # TODO: lrValues = [0.1, 0.01, 0.001, 0.0001]
    def __init__(self, initial_learning_rate=0.001, batch_size=64):
        self.batch_size = batch_size
        self.lr = initial_learning_rate
        self.iTest = 0
        self.get_model()

    # TODO: check which euclidean distance to use
    """
    def euclidean_dist(vect):
    x, y = vect
    sum_square = K.sum(K.square(x-y), axis = 1, keepdims = True)
    result = K.maximum(sum_square, K.epsilon())
    return result
    def l1_dist(vect):
         x, y = vect
         return K.abs(x-y)
    """

    def get_model(self):

        WeightInit1 = RandomNormal(mean=0, stddev=0.01)
        WeightInit2 = RandomNormal(mean=0, stddev=0.01)
        BiasInit = RandomNormal(mean=0.5, stddev=0.01)

        input_shape = (250, 250, 1)
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        # Building the hidden layers of the Network
        ConvolutionNetwork = Sequential()
        ConvolutionNetwork.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape, kernel_initializer=WeightInit1, bias_initializer = BiasInit ,kernel_regularizer=l2(2e-4)))
        ConvolutionNetwork.add(MaxPooling2D())
        ConvolutionNetwork.add(Conv2D(128,(7,7),activation='relu', kernel_initializer=WeightInit1, bias_initializer = BiasInit ,kernel_regularizer=l2(2e-4)))
        ConvolutionNetwork.add(MaxPooling2D())
        ConvolutionNetwork.add(Conv2D(128,(4,4),activation='relu', kernel_initializer=WeightInit1, bias_initializer = BiasInit ,kernel_regularizer=l2(2e-4)))
        ConvolutionNetwork.add(MaxPooling2D())
        ConvolutionNetwork.add(Conv2D(256,(4,4),activation='relu', kernel_initializer=WeightInit1, bias_initializer = BiasInit ,kernel_regularizer=l2(2e-4)))
        ConvolutionNetwork.add(Flatten())
        ConvolutionNetwork.add(Dense(4096,activation="sigmoid", kernel_initializer=WeightInit2, bias_initializer = BiasInit ,kernel_regularizer=l2(1e-3)))

        # Intermediate output of the Siamese Network of left and right
        encoded_l = ConvolutionNetwork(left_input)      # vector sized: 4096 elements, of the left side
        encoded_r = ConvolutionNetwork(right_input)     # vector sized: 4096 elements, of the left side

        # TODO: insert a break point here to see if we are processing a batch of pairs or a single pair
        merge_layer = Lambda(euclidean_dist)([encoded_l, encoded_r])    # merging the two intermediate vectros to a single value using euclidean distance functino

        prediction = Dense(1, activation='sigmoid')(merge_layer)        # the final scalar output of the siamese network, value for each pair in the batch
        self.model = Model(inputs=[left_input, right_input], outputs=prediction)

        optimizer = SGD(lr=0.001, momentum=0.5)     # stochastic gradient descent optimizer

        """
        lr=3e-4, weight_decay=6e-5
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                    initial_learning_rate,
                                                                    decay_steps=10000,
                                                                    decay_rate=0.96,
                    lr_multipliers = {'Conv1': 0.01, 'Conv2':0.01, 'Conv3': 0.01, 'Conv4': 0.01, 'Dense1': 1}
        #opt = Adam(learning_rate = initial_learning_rate)
        opt=  Adam_dlr(lr = initial_learning_rate, lr_multipliers = lr_multipliers)taircase=True)
        """
        #lr_multipliers = {"Conv1": 1, "Conv2":1, "Conv3": 1, "Conv4": 1, "Dense1": 1}
        #opt =  Adam_dlr(learning_rate = 0.00006)
        #opt = SGD(lr = self.lr)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    def getBatchData(self, data, iBatch, sizeBatch):
        startIndex = sizeBatch * iBatch
        #sizeBatch = 1
        X_Batch_Left = []
        X_Batch_Right = []
        y_Batch = []
        for i in range(sizeBatch):
            row_index = startIndex + i
            labelLeft = data.iloc[row_index, 0]
            labelRight = data.iloc[row_index, 2]

            # Similer of Different: True = Similar , False = Different
            if labelRight == labelLeft:
                score = 1
            else:
                score = 0


            # Getting the Left image
            imageNumber1 = getImageNumber(data.iloc[row_index, 1])
            pathLeft = "LFWA\\" + labelLeft + "\\" + labelLeft + imageNumber1
            dataLeft = cv2.imread(pathLeft, cv2.IMREAD_GRAYSCALE)
            dataLeft = dataLeft / 255
            # Getting the Right image
            imageNumber2 = getImageNumber(data.iloc[row_index, 3])
            pathRight = "LFWA\\" + labelRight + "\\" + labelRight + imageNumber2
            dataRight = cv2.imread(pathRight, cv2.IMREAD_GRAYSCALE)
            dataRight = dataRight / 255

            X_Batch_Left.append(np.asarray(dataLeft))
            X_Batch_Right.append(np.asarray(dataRight))
            y_Batch.append(score)

            # plt.imshow(dataRight)
            # plt.gray()
            # plt.show()

            # X_batch = tf.constant(value=[dataLeft, dataRight],dtype='uint8', shape=[250, 250,2 sizeBatch])

        return [np.asarray(X_Batch_Left), np.asarray(X_Batch_Right)], np.expand_dims(np.asarray(y_Batch), -1)


    def training(self, sizeBatch, load_prev_model = False ,best_acc = 0):

        self.val_acc_filename = 'val_acc'
        self.v_acc = []
        self.train_metrics = []
        self.best_acc = best_acc
        self.model_details = {}
        self.model_details['acc'] = 0
        self.model_details['iter'] = 0
        self.model_details['model_lr'] = 0.0
        self.model_details['model_mm'] = 0.0
        linear_inc = 0.01
        self.start = 1
        self.k = 0

        # Don't know what this part does
        # if load_prev_model:
        #     self.continue_training()


        # data_generator = data_gen(self.batch_size, isAug = True)
        # train_generator = data_generator.load_data_batch()

        trainLabels = pd.read_csv("train.csv")
        sizeTraining = trainLabels.shape[0]
        numBatches = sizeTraining // sizeBatch
        trainLabelsShuff = trainLabels.sample(frac=1)
        #trainLabelsShuff = trainLabels


        train_loss, train_acc = [],[]
        i = 0
        start_time = time.time()

        # TODO: while untill we finish epoch

        for iBatch in range(0, numBatches):
            X_Batch, y_batch = self.getBatchData(trainLabelsShuff, iBatch, sizeBatch)  # our version

            # print(X_batch[0].shape,X_batch[1].shape, y_batch.shape)
            # print(type(X_batch), type(y_batch))
            # return

            loss = self.model.train_on_batch(X_Batch, y_batch)
            print(str((iBatch + 1)) + ". Loss = " + str(loss[0]) + " , Accuracy = " + str(loss[1]))
            train_loss.append(loss[0])
            train_acc.append(loss[1])

        totalTime = time.time() - start_time

        avgVecBatchTime = totalTime / numBatches
        print("Average time for a Batch: " + str(avgVecBatchTime))
        print("Time for one Epoch: " + str(totalTime))

        # Update Learning Rate at the end of one Epoch
        K.set_value(self.model.optimizer.learning_rate, K.get_value(self.model.optimizer.learning_rate) * 0.99)
        # maybe update the momentum value in the future
        # K.set_value(self.model.optimizer.momentum,
        #             min(0.9, K.get_value(self.model.optimizer.momentum) + linear_inc))

        return mean(loss)
        # if i % 500 == 0:
        #     train_loss = mean(train_loss)
        #     train_acc = mean(train_acc)
        #     self.train_metrics.append([train_loss, train_acc])
        #
        #     # loss_data.append(loss)
        #
        #     val_acc = self.test_validation_acc(wA_file, uA_file, n_way=20)
        #     # val_acc = [wA_acc, uA_acc]
        #     self.v_acc.append(val_acc)
        #     if val_acc[0] > self.best_acc:
        #         print('\n***Saving model***\n')
        #         # self.model.save_weights("model_{}_val_acc_{}.h5".format(i,val_acc[0]))
        #         self.model.save_weights("best_model/best_model.h5".format(i, val_acc[0]))
        #         self.model_details['acc'] = val_acc[0]
        #         self.model_details['iter'] = i
        #         self.model_details['model_lr'] = K.get_value(self.model.optimizer.learning_rate)
        #         self.model_details['model_mm'] = K.get_value(self.model.optimizer.momentum)
        #         # siamese_net.save(model_path)
        #         self.best_acc = val_acc[0]
        #         with open(self.val_acc_filename, "wb") as f:
        #             pkl.dump((self.v_acc, self.train_metrics), f)
        #         with open('best_model/model_details.pkl', "wb") as f:
        #             pkl.dump(self.model_details, f)
        #
        #     end_time = time.time()
        #     print(
        #         'Iteration :{}  lr :{:.8f} momentum :{:.6f} avg_loss: {:.4f} avg_acc: {:.4f} wA_acc :{:.2f} %  u_Acc: {:.2f} % time_taken {:.2f} s'.format(
        #             i, K.get_value(self.model.optimizer.learning_rate), K.get_value(self.model.optimizer.momentum),
        #             train_loss, train_acc, val_acc[0], val_acc[1], end_time - start_time))
        #
        #     #
        #     train_loss, train_acc = [], []
        #
        # if i % 5000 == 0:
        #     K.set_value(self.model.optimizer.learning_rate, K.get_value(self.model.optimizer.learning_rate) * 0.99)
        #     K.set_value(self.model.optimizer.momentum,
        #                 min(0.9, K.get_value(self.model.optimizer.momentum) + linear_inc))

    def testing(self, sizeBatch):
        testLabels = pd.read_csv("test.csv")
        sizeTesting = testLabels.shape[0]

        X_Batch_Test, y_Batch_Test = self.getBatchData(testLabels, self.iTest, sizeBatch)

        prob = np.asarray(self.model.predict([X_Batch_Test[0], X_Batch_Test[1]]))
        acc = sum(np.round(prob) == y_Batch_Test) / sizeBatch
        # print("prob = " + str(prob))

        # for i in range(sizeBatch):
        #     self.test_one_shot(X_left=X_Batch[0], X_right=X_Batch[1], y=y_Batch)

        return 100*acc
"""
    def train_on_data2(self, load_prev_model = False ,best_acc = 0):

        model_json = self.model.to_json()
        wA_file ='wA_val_10_split_images.pkl'
        uA_file ='uA_val_10_split_images.pkl'


        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        self.val_acc_filename = 'val_acc'

        self.v_acc = []
        self.train_metrics = []
        self.best_acc = best_acc
        self.model_details = {}
        self.model_details['acc'] = 0
        self.model_details['iter'] = 0
        self.model_details['model_lr'] = 0.0
        self.model_details['model_mm'] = 0.0
        linear_inc = 0.01
        self.start = 1
        self.k = 0


        if load_prev_model:
            self.continue_training()

        data_generator = data_gen(self.batch_size, isAug = True)
        train_generator = data_generator.load_data_batch()

        train_loss, train_acc = [],[]
        for i in range(self.start,1000000):


            start_time = time.time()
            X_batch, y_batch = next(train_generator)
            #print(X_batch[0].shape,X_batch[1].shape, y_batch.shape)
            #print(type(X_batch), type(y_batch))
            #return

            loss = self.model.train_on_batch(X_batch, y_batch)
            train_loss.append(loss[0])
            train_acc.append(loss[1])

            if i % 500 == 0:
                train_loss = mean(train_loss)
                train_acc = mean(train_acc)
                self.train_metrics.append([train_loss,train_acc])

                #loss_data.append(loss)

                val_acc  = self.test_validation_acc(wA_file, uA_file, n_way=20)
                #val_acc = [wA_acc, uA_acc]
                self.v_acc.append(val_acc)
                if val_acc[0] > self.best_acc:
                    print('\n***Saving model***\n')
                    #self.model.save_weights("model_{}_val_acc_{}.h5".format(i,val_acc[0]))
                    self.model.save_weights("best_model/best_model.h5".format(i,val_acc[0]))
                    self.model_details['acc'] = val_acc[0]
                    self.model_details['iter'] = i
                    self.model_details['model_lr'] = K.get_value(self.model.optimizer.learning_rate)
                    self.model_details['model_mm'] = K.get_value(self.model.optimizer.momentum)
                    #siamese_net.save(model_path)
                    self.best_acc = val_acc[0]
                    with open(self.val_acc_filename, "wb") as f:
                        pkl.dump((self.v_acc,self.train_metrics), f)
                    with open('best_model/model_details.pkl', "wb") as f:
                        pkl.dump(self.model_details, f)

                end_time = time.time()
                print('Iteration :{}  lr :{:.8f} momentum :{:.6f} avg_loss: {:.4f} avg_acc: {:.4f} wA_acc :{:.2f} %  u_Acc: {:.2f} % time_taken {:.2f} s'.format(i,K.get_value(self.model.optimizer.learning_rate),K.get_value(self.model.optimizer.momentum),train_loss, train_acc,val_acc[0], val_acc[1], end_time-start_time))

                #
                train_loss, train_acc = [],[]

            if i % 5000 == 0:
                K.set_value(self.model.optimizer.learning_rate, K.get_value(self.model.optimizer.learning_rate) * 0.99)
                K.set_value(self.model.optimizer.momentum, min(0.9,K.get_value(self.model.optimizer.momentum) + linear_inc))
"""

def getImageNumber(number):
    if (number // 10 == 0):
        return '_000'+str(number)+'.jpg'
    elif(number // 100 == 0):
        return '_00' + str(number) + '.jpg'
    else:
        return '_0' + str(number) + '.jpg'



if __name__ == "__main__":
    numEpochs = 30
    sizeBatch = 64

    # testLabels = pd.read_csv("test.csv")
    # sizeTest = trainLabels.shape[0]


    model = SiameseNetwork(batch_size=sizeBatch)
    #TODO: while on stopping criria numOfEpoch / not Improving in LOSS

    startEpochTime = time.time()
    for iEpoch in range(numEpochs):
        print("Epoch number: " + str(iEpoch +1))

        # Training phase:
        loss = model.training(sizeBatch)

        # Testing phase:
        if iEpoch % 3 == 0:
            Acc_test = model.testing(sizeBatch)
            print("Test #" + str(model.iTest+1) + ":    Accuracy:" + str(Acc_test))

        # Updating Learning Rate each Epoch (Need to see it's correct)


        # train_loss.append(loss[0])
        # train_acc.append(loss[1])

    totalEpochsTime = time.time() - startEpochTime
    print("Total Epochs Time: " + str(totalEpochsTime))


    """
    training:	1100 pairs
    test: 		500 pairs
    validation: ? (no need for now)
    
    from github
    different sized of training data ( are we traiing once ?)
    30k,90k ..
    evaluate 
    50k training iterations, evaluate model every 500 epochs
    in our terms : 5k training iterations, 
    evaluate model every 50 epocks
    """
