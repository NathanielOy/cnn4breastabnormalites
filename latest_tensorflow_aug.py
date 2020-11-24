# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:33:43 2019

@author: Oyelade
"""

from __future__ import print_function
import imageio
#import matplolib as pyplot
#import matplotlib
#matplotlib.use('Agg')
#%matplotlib inline
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import os
import pandas as pd
from latest_input_fun import input_fn_test, input_fn_train, validation_fn_inputs, train_fn_inputs, get_training_data_old,  \
     plot_images2, plot_images, \
     read_mias_train_data, read_mias_large_image_data, read_mias_test_data,\
     numpy_train_data, numpy_validation_data, numpy_test_data, test_fn_inputs,\
     image_validation_data, image_train_data, image_test_data, getDataSize, getLabels,  prediction_filenames, extract_inbreast_data
"""     
from input_utils import download_file, get_batches, read_and_decode_single_example, load_validation_data, \
    download_data, evaluate_model, get_training_data, load_weights, flatten
"""
from augmentation_op2 import aug_train, aug_validation, aug_test, aug_train_mias1, aug_train_mias2, aug_train_mias3

from pool_helper import PoolHelper
from lrn import LRN
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model


if keras.backend.backend() == 'tensorflow':
    from tensorflow.python.keras import backend as K
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow.python.keras.utils.conv_utils import convert_kernel
    from keras.backend.tensorflow_backend import set_session, clear_session, get_session
    #import multiprocessing as mp
    #mp.set_start_method('spawn', force=True)
    #tf.enable_eager_execution()
    
    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    NUM_PARALLEL_EXEC_UNITS=2
    
    config =  tf.compat.v1.ConfigProto()
    '''
    
                        intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, 
                        inter_op_parallelism_threads=1, 
                        allow_soft_placement=True,
                        log_device_placement=True,
                        device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS}
    '''
    config.gpu_options.per_process_gpu_memory_fraction = 0.333
    session = tf.compat.v1.Session(config=config) #tf.Session(config=config)
    K.set_session(session)
    
    
    
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    #To not use GPU, a good solution is to not allow the environment to see any GPUs by setting the environmental variable CUDA_VISIBLE_DEVICES.
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    #tf.executing_eagerly()

SEED=42
IMG_WIDTH=299  #299  2560, 3328
IMG_HEIGHT=299  #299  2560, 3328
how = "label"
batch_size=16
dataset = 10
if how == "label":
    num_classes = 5
elif how == "normal":
   num_classes = 2
elif how == "mass":
   num_classes = 3
elif how == "benign":
   num_classes = 3
        
def create_cnn_model(weights_path=None):
    # creates our cnn model
    input = Input(shape=(1, IMG_WIDTH, IMG_HEIGHT)) 
    input_pad = ZeroPadding2D(padding=(3, 3))(input)
    conv1_1_3x3_s1 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', name='conv1_1/3x3_s1', kernel_regularizer=l2(0.0002))(input_pad)
    conv1_2_3x3_s1 = Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', name='conv1_2/3x3_s1', kernel_regularizer=l2(0.0002))(conv1_1_3x3_s1) 
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_2_3x3_s1) 
    pool1_helper = PoolHelper()(conv1_zero_pad)
    
    pool1_2_2x2_s1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name='pool1/2x2_s1')(pool1_helper)
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_2_2x2_s1)
    
    conv2_1_3x3_reduce = Conv2D(64, (1,1), padding='same', activation='relu', name='conv2_1/3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_norm1)
    conv2_2_3x3 = Conv2D(64, (3,3), padding='same', activation='relu', name='conv2_2/3x3', kernel_regularizer=l2(0.0002))(conv2_1_3x3_reduce)
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_2_3x3)
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    pool2_helper = PoolHelper()(conv2_zero_pad)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool2/3x3_s2')(pool2_helper)
    
    
    
    
    conv3_1_3x3_s1 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu', name='conv3_1/3x3_s1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    conv3_2_3x3_s1 = Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu', name='conv3_2/3x3_s1', kernel_regularizer=l2(0.0002))(conv3_1_3x3_s1)
    conv3_zero_pad = ZeroPadding2D(padding=(1, 1))(conv3_2_3x3_s1)
    pool3_helper = PoolHelper()(conv3_zero_pad)
    pool3_2_2x2_s1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name='pool3/2x2_s1')(pool3_helper)
    pool3_norm1 = LRN(name='pool3/norm1')(pool3_2_2x2_s1)

    conv4_1_3x3_reduce = Conv2D(256, (1,1), padding='same', activation='relu', name='conv4_1/3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_norm1)
    conv4_2_3x3 = Conv2D(256, (3,3), padding='same', activation='relu', name='conv4_2/3x3', kernel_regularizer=l2(0.0002))(conv4_1_3x3_reduce)
    conv4_norm2 = LRN(name='conv4/norm2')(conv4_2_3x3)
    conv4_zero_pad = ZeroPadding2D(padding=(1, 1))(conv4_norm2)
    pool4_helper = PoolHelper()(conv4_zero_pad)
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool4/3x3_s2')(pool4_helper)
    
     
    
    conv5_1_3x3_s1 = Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', name='conv5_1/3x3_s1', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    conv5_2_3x3_s1 = Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu', name='conv5_2/3x3_s1', kernel_regularizer=l2(0.0002))(conv5_1_3x3_s1)
    conv5_zero_pad = ZeroPadding2D(padding=(1, 1))(conv5_2_3x3_s1)
    pool5_helper = PoolHelper()(conv5_zero_pad)
    pool5_2_2x2_s1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same', name='pool5/2x2_s1')(pool5_helper)
    pool5_norm1 = LRN(name='pool5/norm1')(pool5_2_2x2_s1)

    conv6_1_3x3_reduce = Conv2D(1024, (1,1), padding='same', activation='relu', name='conv6_1/3x3_reduce', kernel_regularizer=l2(0.0002))(pool5_norm1)
    conv6_2_3x3 = Conv2D(1024, (3,3), padding='same', activation='relu', name='conv6_2/3x3', kernel_regularizer=l2(0.0002))(conv6_1_3x3_reduce)
    conv6_norm2 = LRN(name='conv6/norm2')(conv6_2_3x3)
    conv6_zero_pad = ZeroPadding2D(padding=(1, 1))(conv6_norm2)
    pool6_helper = PoolHelper()(conv6_zero_pad)
    pool6_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool6/3x3_s2')(pool6_helper)
    
    
    pool7_2x2_s1 = AveragePooling2D(pool_size=(2,2), strides=(1,1), name='pool7/2x2_s1')(pool6_3x3_s2)
    
    loss_flat = Flatten()(pool7_2x2_s1)
    pool7_drop_2x2_s1 = Dropout(rate=0.5)(loss_flat)
    loss_classifier = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002))(pool7_drop_2x2_s1)
    loss_classifier_act = Activation('softmax', name='prob')(loss_classifier)

    mynet = Model(inputs=input, outputs=[loss_classifier_act])

    if weights_path:
        mynet.load_weights(weights_path)

    if keras.backend.backend() == 'tensorflow':
        # convert the convolutional kernels for tensorflow
        ops = []
        for layer in mynet.layers:
            if layer.__class__.__name__ == 'Conv2D':
                original_w = K.get_value(layer.kernel)
                converted_w = convert_kernel(original_w)
                ops.append(tf.compat.v1.assign(layer.kernel, converted_w).op)
        K.get_session().run(ops)

    return mynet
    
    
    
if __name__ == "__main__":

    #validation_fn_inputs(K.get_session(), epochs, batch_size, None) 
    #extract_inbreast_data()
    #read_mias_large_image_data()
    #read_mias_test_data()
    #read_mias_train_data()
    #train_dataset=train_fn_inputs(K.get_session(), 10, batch_size, None) 
    #val_data=validation_fn_inputs(K.get_session(), 10, batch_size, None) 
    #data_dir1 = './AugTrain/'
    #data_dir2 = './gen/'
    #data_dir3 = './miaswhole/'
    #plot_images2(data_dir1, 'AugTrain')
    #plot_images2(data_dir2, '')
    #plot_images2(data_dir3, '')
    """
    train_files, total_records = get_training_data_old(what=dataset)
    steps_per_epoch = int(total_records / batch_size)
    print("Steps per epoch:", steps_per_epoch)
    image, label = read_and_decode_single_example(train_files, label_type=how, normalize=False)
    X_def, y_def = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=2000,min_after_dequeue=1000)   #enqueue_many=True
    """
    # Test pretrained model
    model = create_cnn_model()  #'googlenet_weights.h5'
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) #epsilon=1e-08, for Keras; epsilon=1e-08 for tensorflow; epsilon=1e-8 for Tocrch or MxNet 
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'],) 
    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath="weights-{epoch:02d}-{loss:.4f}.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') every improvement in accuracy
    #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min') every impromnet in loss
    checkpoint=ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1) # every epoch
    
    callbacks_list = [checkpoint] #early_stopping
    trainaug=aug_train_mias3(batch_size)  #aug_train_mias2
    valaug=aug_validation(batch_size)
    testaug=aug_test(batch_size)
    
    epochs=100
    
    dataWant='numpy_only' #numpy_mias, inbreast, numpy_only, large_mias
    if dataWant == "large_mias":
        dWhat=0
    elif dataWant == "numpy_mias":
       dWhat=1
    elif dataWant == "numpy_only":
       dWhat=3
    elif dataWant == "inbreast":
       dWhat=2
    elif dataWant == "ddsm":
        dWhat=-1
    
    if dWhat==-1:
        train_dataset=train_fn_inputs(K.get_session(), epochs, batch_size, trainaug) #numpy_train_data(K.get_session(), batch_size)#, 
        val_data=validation_fn_inputs(K.get_session(), epochs, batch_size, valaug) #numpy_validation_data(K.get_session(), batch_size)#, 
        test_dataset =test_fn_inputs(K.get_session(), epochs, batch_size, testaug) #numpy_test_data(K.get_session(), batch_size)#, test_fn_inputs(K.get_session(), epochs, batch_size, None) 
    else:
        train_dataset=numpy_train_data(K.get_session(), batch_size, trainaug)#, train_fn_inputs(K.get_session(), epochs, batch_size, trainaug) #
        val_data=numpy_validation_data(K.get_session(), batch_size, valaug)#, validation_fn_inputs(K.get_session(), epochs, batch_size, valaug) #
        test_dataset =numpy_test_data(K.get_session(), batch_size, testaug)#, test_fn_inputs(K.get_session(), epochs, batch_size, None) test_fn_inputs(K.get_session(), epochs, batch_size, testaug) #
    """
    train_dataset=trainaug.flow_from_directory(
                    directory=r"./train/",
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    color_mode="grayscale",
                    batch_size=batch_size,
                    class_mode="categorical",
                    shuffle=True,
                    seed=SEED
                  )#train_fn_inputs(K.get_session(), epochs, batch_size, trainaug) 
    val_data=valaug.flow_from_directory(
                directory=r"./val/",
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                color_mode="grayscale",
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True,
                seed=SEED
            )#validation_fn_inputs(K.get_session(), epochs, batch_size, valaug) 
    test_dataset=testaug.flow_from_directory(
                directory=r"./test/",
                target_size=(IMG_WIDTH, IMG_HEIGHT),
                color_mode="grayscale",
                batch_size=batch_size,
                class_mode=None,
                shuffle=False,
                seed=SEED
            ) 
    """
      
    if dWhat==-1:
        total_records = 44712  
        val_records = 11178  
        TRAIN_STEPS_PER_EPOCH=int(total_records // batch_size)
    else:
        train_records, val_records=getDataSize(1, dWhat)
        TRAIN_STEPS_PER_EPOCH=int(train_records // batch_size)
        val_records=int(val_records)
    
    hist=model.fit_generator(train_dataset,
                   steps_per_epoch=TRAIN_STEPS_PER_EPOCH, #(training_df.shape[0])//batchsize,
                   epochs=epochs,
                   verbose = 1,
                   callbacks=callbacks_list,
                   validation_data=val_data, 
                   validation_steps=val_records//batch_size,
                   workers=0
              )
    
    mydir='/content/'  #'./'    #'./'  '/content/gdrive/My Drive/'
    print(hist.history)
    
    model_path1 = os.path.join(mydir, "my_cnn_weights.h5")
    model_path2 = os.path.join(mydir, "my_cnn_model.h5")
    model.save(model_path2)
    model.save_weights(model_path1)
    
    #model.summary()
    tf.keras.utils.plot_model(model, to_file=mydir+'archi_distortion_model1.png')
    tf.keras.utils.plot_model(model, to_file=mydir+'archi_distortion_model2.png', show_shapes=True, show_layer_names=True)
    
    with open(mydir+'output.txt', 'w') as f:
        f.write(str(hist.history['loss']))
        f.write(str(hist.history['val_loss']))
        f.write(str(hist.history['acc']))
        f.write(str(hist.history))
    
    
    print("Training Loss: ", hist.history['loss'])
    print("Validation Loss: ", hist.history['val_loss'])
    print("Training Accuracy: ", hist.history['acc'])
    print("Training Accuracy: ", hist.history['val_acc'])
    
  
    N = epochs
    plt.style.use('seaborn-whitegrid')
    # plot training history, np.arange(0, N),  where N-epochs  plt.figure()
    plt.title("Training/Validation Loss on Dataset "+dataWant)
    plt.plot(np.arange(0, N),hist.history['loss'], label='training')
    plt.plot(np.arange(0, N),hist.history['val_loss'], label='validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.show()
    
    plt.title("Training/Validation Accuracy on Dataset"+dataWant)
    plt.plot(np.arange(0, N),hist.history["acc"], label="train_accuracy")
    plt.plot(np.arange(0, N),hist.history["val_acc"], label="validation_accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch #")
    plt.legend(['training', 'validation'], loc="lower left")
    plt.show()
    
    if dWhat == -1:
        score=model.evaluate_generator(validation_fn_inputs, 
                       steps=None, 
                       max_queue_size=10,
                       verbose = 1,
                       workers=1, 
                       use_multiprocessing=True)
    else:
        val_records, _=getDataSize(2, dWhat)
        STEP_SIZE_VALID=int(val_records // batch_size)
        score=model.evaluate(val_data, steps=STEP_SIZE_VALID, verbose = 1,) #evaluate_generator(val_data, steps=STEP_SIZE_VALID, verbose = 1,)
        print("Loss Val: ", score[0], "Accuracy Val: ", score[1])
    
    
    plt.title("Evaluation Loss and Accuracy on Dataset")
    plt.plot(score[0], label='Eval-Loss')
    plt.plot(score[1], label="Eval Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(['validation loss', 'validation acc'], loc="lower left")
    plt.show();
    
    
    test_records, _=getDataSize(3, dWhat)
    STEP_SIZE_TEST=int(test_records // batch_size)
    my_model = tf.keras.models.load_model('/content/my_cnn_model.h5')
    pred=my_model.predict(test_dataset,steps=STEP_SIZE_TEST,verbose=1)#predict_generator(test_dataset,steps=STEP_SIZE_TEST,verbose=1)
    print(pred)
    
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(pred[0], label='Test-Loss')
    plt.plot(pred[1], label="Test Accuracy")
    plt.title("Test Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show();
    
    predicted_class_indices=np.argmax(pred,axis=1)
    print(predicted_class_indices)
    
    labels = getLabels()#(train_dataset.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames=prediction_filenames(dWhat)
    results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
    results.to_csv("results.csv",index=False)
    
    '''
    # predict probabilities for test set
yhat_probs = model.predict(testX, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(testX, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(testy, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(testy, yhat_classes)
print(matrix)



# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# generate and prepare the dataset
def get_data():
	# generate dataset
	X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
	# split into train and test
	n_test = 500
	trainX, testX = X[:n_test, :], X[n_test:, :]
	trainy, testy = y[:n_test], y[n_test:]
	return trainX, trainy, testX, testy

# define and fit the model
def get_model(trainX, trainy):
	# define model
	model = Sequential()
	model.add(Dense(100, input_dim=2, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=300, verbose=0)
	return model

# generate data
trainX, trainy, testX, testy = get_data()
# fit model
model = get_model(trainX, trainy)


# predict probabilities for test set
yhat_probs = model.predict(testX, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(testX, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(testy, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(testy, yhat_classes)
print(matrix)
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
    '''
    
   