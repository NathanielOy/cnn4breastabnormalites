# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:54:31 2019

@author: Oyelade
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
#tf.enable_eager_execution()
import os
import numpy as np
#import imageio
from PIL import Image
from math import floor
import pathlib
import PIL 
import IPython.display as display
import shutil
#import tensorflow.contrib.eager as tfe
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import csv
import cv2
"""
'images and labels in tfrecords format'
train_path_10 = os.path.join( "..", "Dataset", "ddsm-mammography", "training10_0", "training10_0.tfrecords")
train_path_11 = os.path.join( "..", "Dataset", "ddsm-mammography", "training10_1","training10_1.tfrecords")
train_path_12 = os.path.join( "..", "Dataset", "ddsm-mammography", "training10_2", "training10_2.tfrecords")
train_path_13 = os.path.join( "..", "Dataset",  "ddsm-mammography", "training10_3", "training10_3.tfrecords")
train_path_14 = os.path.join( "..", "Dataset",  "ddsm-mammography", "training10_4", "training10_4.tfrecords")

'mias 299X299 numpy files'
all_mias_labels9 = os.path.join( "..", "Dataset",  "mias", "train", "all_mias_labels9.npy")
all_mias_slices9 = os.path.join( "..", "Dataset",  "mias", "train", "all_mias_slices9.npy")
mias_test_labels_enc= os.path.join( "..", "Dataset",  "mias", "test", "test12_labels.npy")
mias_test_images = os.path.join( "..", "Dataset",  "mias", "test", "test12_data.npy")
mias_val_labels_enc= os.path.join( "..", "Dataset",  "mias", "val", "all_mias_labels.npy")
mias_val_images = os.path.join( "..", "Dataset",  "mias", "val", "all_mias_slices.npy")

'Images and labels of MIAS and INbreast arranged in the format of aug.flow_from_directory'
mias_numpy_train = os.path.join( "..", "Dataset",  "images", "data", "train")
mias_large_train = os.path.join( "..", "Dataset",  "images", "data2", "train")
inbreast_train = os.path.join( "..", "Dataset",  "images", "data3", "train")
mias_numpy_val = os.path.join( "..", "Dataset",  "images", "data", "valid")
mias_large_val = os.path.join( "..", "Dataset",  "images", "data2", "valid")
inbreast_val = os.path.join( "..", "Dataset",  "images", "data3", "valid")
mias_numpy_test = os.path.join( "..", "Dataset",  "images", "data", "test")
mias_large_test = os.path.join( "..", "Dataset",  "images", "data2", "test")
inbreast_test = os.path.join( "..", "Dataset",  "images", "data3", "test")


'images and labels of the mias and Inbreast in size 1024X1024'
mias_large_images = os.path.join( "..", "Dataset",  "mias", "train", "all-mias")
mias_large_images_labels = os.path.join( "..", "Dataset",  "mias", "train", "all-mias", "Info.txt")
inbreast_large_images = os.path.join( "..", "Dataset",  "inbreast", "train", "all-mias")
inbreast_large_images_labels = os.path.join( "..", "Dataset",  "inbreast", "train", "all-mias", "Info.txt")

train_path_10 = "/mdata/training10_0/training10_0.tfrecords" #mdata/training10_     content
train_path_11 = "/mdata/training10_1/training10_1.tfrecords"
train_path_12 = "/mdata/training10_2/training10_2.tfrecords"
train_path_13 = "/mdata/training10_3/training10_3.tfrecords"
train_path_14 = "/mdata/training10_4/training10_4.tfrecords"

"""


train_path_10 = "/content/training10_0.tfrecords" #mdata/training10_     content
train_path_11 = "/content/training10_1.tfrecords"
train_path_12 = "/content/training10_2.tfrecords"
train_path_13 = "/content/training10_3.tfrecords"
train_path_14 = "/content/training10_4.tfrecords"



all_mias_labels9 = "/content/all_mias_labels9.npy"   #mias/train/all_mias_labels9.npy
all_mias_slices9 = "/content/all_mias_slices9.npy"   #mias/train/all_mias_slices9.npy
mias_test_labels_enc= "/content/test12_labels.npy"   #mias/test/test12_labels.npy
mias_test_images = "/content/test12_data.npy"        #mias/test/test12_data.npy
mias_test_labels_enc2= "/content/test10_labels.npy"   #mias/test/test12_labels.npy
mias_test_images2 = "/content/test10_data.npy"        #mias/test/test12_data.npy
mias_val_labels_enc= "/content/all_mias_labels.npy"  #mias/val/all_mias_labels.npy
mias_val_images = "/content/all_mias_slices.npy"     #mias/val/all_mias_slices.npy

'Images and labels of MIAS and INbreast arranged in the format of aug.flow_from_directory'
mias_numpy_train = "/content/data/train"
mias_large_train = "/content/data2/train"
inbreast_train = "/content/data3/train"
mias_numpy_val = "/content/data/valid"
mias_large_val = "/content/data2/valid"
inbreast_val = "/content/data3/valid"
mias_numpy_test = "/content/data/test"
mias_large_test = "/content/data2/test"
inbreast_test = "/content/data3/test"


'images and labels of the mias and Inbreast in size 1024X1024'
mias_large_images = "/mdata/mias/train/all-mias"
mias_large_images_labels = "/mdata/mias/train/all-mias/Info.txt"
inbreast_large_images = "/mdata/inbreast/train/all-mias"
inbreast_large_images_labels = "/mdata/inbreast/train/all-mias/Info.txt"

  
IMG_WIDTH=299 #299   1024 2560
IMG_HEIGHT=299 #299 1024  3328
NUM_CLASSES = 5
batch_size=128
buf_size=10000
AUTOTUNE=tf.data.experimental.AUTOTUNE
CLASS_NAMES=['NORM', 'CIRC', 'MISC', 'ASYM', 'ARCH',  'SPIC'] #'CALC',

## Load the training data and return a list of the tfrecords file and the size of the dataset
## Multiple data sets have been created for this project, which one to be used can be set with the type argument
def getLabels():
    return CLASS_NAMES

def getDataSize(dType, dWhat):
    image_count=0
    val_count=0
    
    if dType ==1: #train data
        data=get_mias_image_training_data()[dWhat]
        val=get_mias_image_validation_data()[dWhat]
        if dWhat == 3: # for numpy files and not image files like .png 
           _images = np.load(val)
           val_count=_images.shape[0]
        else:
           val_dir = pathlib.Path(val)
           val_count = len(list(val_dir.glob('*/*.png')))
        
    if dType ==2: #validation data
         data=get_mias_image_validation_data()[dWhat]
         
    if dType ==3: #test data
         data=get_mias_image_test_data()[dWhat]
    
    if dWhat == 3: # for numpy files and not image files like .png 
        _images = np.load(data)
        image_count=_images.shape[0]
    else: 
        data_dir = pathlib.Path(data)
        image_count = len(list(data_dir.glob('*/*.png')))
        print(image_count)
        print([item for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    
    return image_count, val_count

def prediction_filenames(dWhat):
    data=get_mias_image_test_data()[dWhat]
    data_dir = pathlib.Path(data)
    filenames=[item for item in data_dir.glob(data_dir.glob('*/*.png')) if item.name != "LICENSE.txt"]
    return filenames
    
def get_mias_image_training_data():
    train_files = [mias_large_train, mias_numpy_train, inbreast_train, all_mias_slices9] #
    return train_files

def get_mias_image_validation_data():
    test_files = [mias_large_val, mias_numpy_val, inbreast_val, mias_val_images] 
    return test_files

def get_mias_image_test_data():
    test_files = [mias_large_test, mias_numpy_test, inbreast_test, mias_test_images] #
    return test_files

def get_mias_numpy_training_data():
    train_files = [all_mias_slices9, all_mias_labels9]
    return train_files

def get_mias_numpy_test_data():
    test_files = [mias_test_images2, mias_test_labels_enc2]
    return test_files

def get_mias_numpy_validation_data():
    validation_files = [mias_val_images, mias_val_labels_enc]
    return validation_files


'getting images prepared as dataset format'
def get_label(file_path):
  parts = tf.strings.split(file_path, '/') # convert the path to a list of path components
  parts=tf.strings.split(parts[-1], '.')
  parts=tf.strings.split(parts[-2], '_')
  # The index last is the class-directory
  return parts[-1] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=1)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])  #[1, IMG_WIDTH, IMG_HEIGHT]

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def prepare_for_training(bs, ds, cache=False, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.  
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  ds = ds.shuffle(buffer_size=shuffle_buffer_size).repeat().batch(bs).prefetch(bs)
  return ds

def image_train_data(sess, bs):
    train_image = get_mias_image_training_data()
    data_dir = pathlib.Path(train_image[0])
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*')) #train_image[0]+'\*.png' list(data_dir.glob('*/*.png'))
    for f in list_ds.take(1):
        print(f)
        
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    train_ds = prepare_for_training(bs, labeled_ds)
    train_ds=train_ds.make_initializable_iterator()  
    image, label = train_ds.get_next()
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    sess.run(train_ds.initializer)
    while True:
        yield image, label
    #image_batch, label_batch = next(iter(train_ds))
    #yield image_batch, label_batch

def image_validation_data(sess, bs):
    val_files = get_mias_image_validation_data()
    data_dir = pathlib.Path(val_files[0])
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*')) #train_image[0]+'\*.png' list(data_dir.glob('*/*.png'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = prepare_for_training(bs, labeled_ds)
    val_ds=val_ds.make_initializable_iterator()  
    image, label = val_ds.get_next()
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    sess.run(val_ds.initializer)
    while True:
        yield image, label  
    
def image_test_data(sess, bs):
    test_files = get_mias_image_test_data()
    data_dir = pathlib.Path(test_files[0])
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*')) #train_image[0]+'\*.png' list(data_dir.glob('*/*.png'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    test_ds = prepare_for_training(bs, labeled_ds)
    test_ds=test_ds.make_initializable_iterator()  
    image, label = test_ds.get_next()
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    sess.run(test_ds.initializer)
    while True:
        yield image, label

'Read all numpy files in dataset for training'
def numpy_train_data(sess, bs,  aug=None):
    folder_path = './TrainingAugImages'
    shutil.rmtree(folder_path, ignore_errors = True)
    os.mkdir(folder_path)
    train_files = get_mias_numpy_training_data()
    train_image = np.load(train_files[0])
    train_labels = np.load(train_files[1])
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
    train_dataset = train_dataset.shuffle(buf_size).batch(bs)
    train_dataset=tf.compat.v1.data.make_initializable_iterator(train_dataset)  
    image, label = train_dataset.get_next()
    label = tf.one_hot(label, NUM_CLASSES)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    print(image.shape)
    sess.run(train_dataset.initializer)
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    
    while True:
        #if aug != None:
        #image=next((aug.flow(image, batch_size=bs, save_to_dir='./TrainingAugImages', save_prefix='aug', save_format='png')))
        yield image, label

def numpy_validation_data(sess, bs,  aug=None):
    folder_path = './ValidationAugImages'
    shutil.rmtree(folder_path, ignore_errors = True)
    os.mkdir(folder_path)
    
    validation_files = get_mias_numpy_validation_data()
    validation_image = np.load(validation_files[0])
    validation_labels = np.load(validation_files[1])
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_image, validation_labels))
    validation_dataset = validation_dataset.batch(bs) #.make_initializable_iterator()  
    validation_dataset=tf.compat.v1.data.make_initializable_iterator(validation_dataset)  
    image, label = validation_dataset.get_next()
    label = tf.one_hot(label, NUM_CLASSES)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    print(image.shape)
    sess.run(validation_dataset.initializer)
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    
    while True:
        #if aug != None:
        #image=next((aug.flow(image, batch_size=bs, save_to_dir='./ValidationAugImages', save_prefix='aug', save_format='png')))
        yield image, label

def numpy_test_data(sess, bs,  aug=None):
    folder_path = './TestAugImages'
    shutil.rmtree(folder_path, ignore_errors = True)
    os.mkdir(folder_path)
    
    test_files = get_mias_numpy_test_data()
    test_image = np.load(test_files[0])
    test_labels = np.load(test_files[1])
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
    test_dataset = test_dataset.batch(bs) #.make_initializable_iterator()  
    test_dataset=tf.compat.v1.data.make_initializable_iterator(test_dataset)  
    image, label = test_dataset.get_next()
    label = tf.one_hot(label, NUM_CLASSES)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    print(image.shape)
    with tf.compat.v1.Session() as sess:
        sess.run(test_dataset.initializer)
        image=image.eval(session=sess)
        label=label.eval(session=sess)
    
        while True:
            #if aug != None:
            #image=next((aug.flow(image, batch_size=bs, save_to_dir='./TestAugImages', save_prefix='aug', save_format='png')))
            yield image, label



'INbreast data extraction'
def extract_inbreast_data():
    PNG = True # make it True if you want in PNG format
    folder_path = "../Dataset/Inbreast/AllDICOMs/"  # Specify the .dcm folder path  AllROI
    jpg_folder_path = "./inbreast/"  # Specify the .jpg/.png folder path
    folder_path2='./data3/train/'
    images_path = os.listdir(folder_path)
    
    batches = 0
    max_batches = 6
    img_gen = []
    label_gen = []
    label_normal_gen = []
    
    dicom_image_description = pd.read_csv("../Dataset/Inbreast/INbreast2.csv",
         sep=',',           # Tab-separated value file.
         dtype={"File Name": str},
         usecols=['File Name',	'Mass', 'Micros',	'Distortion',	'Asymmetry']   # Only load the three columns specified.
                                         )
    with open('Patient_Detail.csv', 'w', newline ='') as csvfile:
        fieldnames = ['File Name',	'Mass', 'Micros',	'Distortion',	'Asymmetry'] #list(dicom_image_description["File Name"])
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fieldnames)
        for n, image in enumerate(images_path):
            ds = dicom.dcmread(os.path.join(folder_path, image))
            pixel_array_numpy = ds.pixel_array            
            image_number=image.split('_')[0]
            image_view=image.split('_')[4] #image view e.g MLO
            image_which=image.split('_')[3] #left or right breast
            #print(image_number)
            #print(dicom_image_description["File Name"])
            
            i=0
            for image_index in list(dicom_image_description["File Name"]):
                if image_number == image_index:
                    mass=dicom_image_description["Mass"][i]
                    micros=dicom_image_description["Micros"][i]
                    distortion=dicom_image_description["Distortion"][i]
                    asymmetry=dicom_image_description["Asymmetry"][i]
                    CLASS_NAMES=['NORM', 'CIRC', 'MISC', 'ASYM', 'ARCH', 'CALC', 'SPIC']
                    im = Image.fromarray(pixel_array_numpy)
                    im = im.convert("L")
                    label=''
                    
                    if mass == 'X':
                        im.save(folder_path2+"/"+CLASS_NAMES[2]+"/"+CLASS_NAMES[2]+"_"+'{}'.format(i)+".png")
                        label+='_'+CLASS_NAMES[2]
                    if micros == 'X':
                        im.save(folder_path2+"/"+CLASS_NAMES[1]+"/"+CLASS_NAMES[1]+"_"+'{}'.format(i)+".png")
                        label+='_'+CLASS_NAMES[1]
                    if distortion == 'X':
                        im.save(folder_path2+"/"+CLASS_NAMES[4]+"/"+CLASS_NAMES[4]+"_"+'{}'.format(i)+".png")
                        label+='_'+CLASS_NAMES[4]
                    if asymmetry == 'X':
                        im.save(folder_path2+"/"+CLASS_NAMES[3]+"/"+CLASS_NAMES[3]+"_"+'{}'.format(i)+".png")
                        label+='_'+CLASS_NAMES[3]
                    if mass != 'X' and micros != 'X' and distortion != 'X' and asymmetry != 'X':
                        im.save(folder_path2+"/"+CLASS_NAMES[0]+"/"+CLASS_NAMES[0]+"_"+'{}'.format(i)+".png")
                        label+='_'+CLASS_NAMES[0]
                    
                    label+='_'+image_view+image_which
                    if batches <= max_batches and CLASS_NAMES[4] in label:
                        img_gen.append(im)
                        label_normal_gen.append(label)
                        label_gen.append('')
                        batches += 1
                    
                i+=1
            
            if PNG == False:
                image = image.replace('.dcm', '.jpg')
            else:
                image = image.replace('.dcm', '.png')
            im.save(os.path.join(jpg_folder_path, image))
            #cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
            if n % 50 == 0:
                print('{} image converted'.format(n))
            '''
            rows = []
            for field in fieldnames:
                if ds.data_element(field) is None:
                    rows.append('')
                else:
                    x = str(ds.data_element(field)).replace("'", "")
                    y = x.find(":")
                    x = x[y+2:]
                    rows.append(x)
            writer.writerow(rows)
            '''
    plot_images(img_gen, label_gen, label_normal_gen, 'INbreast',  rows=1, figsize=(20,16))
        
'Extracting large images from files'    
def read_mias_large_image_data():
    folder_path='./miaswhole'
    folder_path2='./data2/train'
    
    batches = 0
    max_batches = 6
    img_gen = []
    label_gen = []
    label_normal_gen = []
    lines=[]
    i=0
    with open(mias_large_images_labels, 'r') as data:
        lines = [line.rstrip('\n') for line in data]
    
    for file in os.listdir(mias_large_images):
        if file.endswith(".pgm"):
            index=os.path.splitext(file)[0]
            for line in lines:
                element=line.split(' ')
                print(element)
                for elm in element:
                    if elm==index:
                        print(file+"_"+index+"_"+line)
                        im = np.array(Image.open(mias_large_images+"/"+file))
                        im=Image.fromarray(im)
                        im = im.convert("L")
                        label_normal=line.split(' ')[2]
                        im.save(folder_path+"/"+index+"_"+label_normal+".png")
                        im.save(folder_path2+"/"+label_normal+"/"+label_normal+"_"+'{}'.format(i)+".png")
                        batches += 1
                        if batches <= max_batches:
                            img_gen.append(im)
                            label_normal_gen.append(label_normal)
                            label_gen.append('')
                        break;     
        i+=1
    #plot_images(img_gen, label_gen, label_normal_gen, '',  rows=1, figsize=(20,16))                   
                
def read_mias_train_data():
    #start=time.time()
    train_files = get_mias_numpy_training_data()
    image = np.load(train_files[0])
    label_normal = np.load(train_files[1])
    #end=time.time()
    print("\nData summary:\n", image)
    print("\nData shape:\n", image.shape)
    print("\nData summary:\n", label_normal)
    print("\nData shape:\n", label_normal.shape)
    image = np.reshape(image, [image.shape[0], 1, IMG_WIDTH, IMG_HEIGHT])
    batches = 0
    max_batches = 3075
    img_gen = []
    label_gen = []
    label_normal_gen = []
    folder_path='./miasTrain'
    folder_path2='./data/train'
        
    for i in range(image.shape[0]):
        img_gen.append(image[i,0,:,:])
        
        im = Image.fromarray(image[i,0,:,:])
        im = im.convert("L")
        im.save(folder_path+"/"+"img_"+'{}'.format(i)+"_"+CLASS_NAMES[label_normal[i]]+".png")
        im.save(folder_path2+"/"+CLASS_NAMES[label_normal[i]]+"/"+CLASS_NAMES[label_normal[i]]+"_"+'{}'.format(i)+".png")
        label_normal_gen.append(label_normal[i].astype('U13'))
        label_gen.append('')
        
        batches += 1
        if batches >= max_batches:
            break
    plot_images(img_gen, label_gen, label_normal_gen, 'MIAS',  rows=1, figsize=(20,16))

#get_mias_numpy_validation_data
def read_mias_test_data():
    test_files = get_mias_numpy_test_data()
    image = np.load(test_files[0])
    label_normal = np.load(test_files[1])
    #end=time.time()
    print("\nData summary:\n", image)
    print("\nData shape:\n", image.shape)
    print("\nData summary:\n", label_normal)
    print("\nData shape:\n", label_normal.shape)
    image = np.reshape(image, [image.shape[0], 1, IMG_WIDTH, IMG_HEIGHT])
    batches = 0
    max_batches = 6
    img_gen = []
    label_gen = []
    label_normal_gen = []
    folder_path='./miasValidate'
    folder_path2='./data/test'
        
    for i in range(image.shape[0]):
        img_gen.append(image[i,0,:,:])
        
        im = Image.fromarray(image[i,0,:,:])
        im = im.convert("L")
        im.save(folder_path+"/"+"img_"+'{}'.format(i)+"_"+label_normal[i].astype('U13')+".png")
        im.save(folder_path2+"/"+CLASS_NAMES[label_normal[i]]+"/"+CLASS_NAMES[label_normal[i]]+"_"+'{}'.format(i)+".png")
        label_normal_gen.append(label_normal[i].astype('U13'))
        label_gen.append('')
        
        batches += 1
        if batches >= max_batches:
            break
    plot_images(img_gen, label_gen, label_normal_gen, 'MIAS',  rows=1, figsize=(20,16))

 
def get_training_data_old(what=10):
    if what == 10:
        train_files = [train_path_10, train_path_11, train_path_12, train_path_13]
        total_records = 55890
    else:
        raise ValueError('Invalid dataset!')

    return train_files, total_records

def get_test_data(what=10):
    test_files = [train_path_14]
    
    return [test_files], 11178
    

def _parse_image_function(tfrecord):
        image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
        }
        
        image_feature_description2 = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        }
        # Parse the input tf.Example proto using the dictionary above.    image2 =tf.decode_raw(sample['image'], tf.uint8)
        #features=tf.io.parse_single_example(proto, image_feature_description2) #image1 = tf.image.decode_bmp(sample['image'], channels=3)
        
        sample=tf.io.parse_single_example(tfrecord, image_feature_description2) 
        image =tf.io.decode_raw(sample['image'], tf.uint8)
        #image = tf.image.decode_jpeg(sample['image'], channels=3)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        #image = tf.image.resize_bilinear(image, [3, 299, 299], align_corners=False)
        image = tf.reshape(image, [1, IMG_WIDTH, IMG_HEIGHT])
        #image = tf.image.resize_image_with_crop_or_pad(image, [3, IMG_WIDTH, IMG_HEIGHT])
        label = tf.cast(sample["label"], tf.int32)
        image = tf.cast(image, tf.float32)
        label_normal = tf.cast(sample["label_normal"], tf.int32) 
        filename=2;
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label#, label_normal]  #, filename 
def extract_tfrecords(sess, num_images, image,label, folder_path):
    
    image_raw = image
    for i in range(num_images):
        im_,lbl = sess.run([image,label])
        print(im_.shape)
        #label=lbl.decode('utf-8')
        #lbl_=lbl.decode("utf-8")
        nlabel=np.char.decode(lbl.astype(np.bytes_), 'UTF-8')
        img_name = '{}.png'.format(i + 1)
        savePath = os.path.join(folder_path, img_name)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        mpimg.imsave(savePath, im_, cmap='Greys_r', vmin=0, vmax=1)
        display.display(display.Image(data=image_raw))

def validation_fn_inputs(sess, epochs, bs,  aug=None):
    test_files, total_records = get_test_data()
    steps_per_epoch = int(total_records / batch_size)
    folder_path = './ValidationAugImages'
    shutil.rmtree(folder_path, ignore_errors = True)
    os.mkdir(folder_path)
    raw_dataset = tf.data.TFRecordDataset(test_files)
    parsed_image_dataset = raw_dataset.map(_parse_image_function).shuffle(buffer_size=buf_size).batch(batch_size)#.make_initializable_iterator() 
    parsed_image_dataset=tf.compat.v1.data.make_initializable_iterator(parsed_image_dataset)  
    image, label = parsed_image_dataset.get_next()
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    sess.run(parsed_image_dataset.initializer)
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    while True:
        #image=next((aug.flow(image, batch_size=bs, save_to_dir='./ValidationAugImages', save_prefix='aug', save_format='png')))
        yield image, label

def train_fn_inputs(sess, epochs, bs,  aug=None):
    train_files, total_records = get_training_data_old()
    steps_per_epoch = int(total_records / batch_size)    
    folder_path = './TrainingAugImages'
    shutil.rmtree(folder_path, ignore_errors = True)
    os.mkdir(folder_path)
    raw_dataset = tf.data.TFRecordDataset(train_files)     #.repeat() .repeat(epochs)
    parsed_image_dataset = raw_dataset.map(_parse_image_function).shuffle(buffer_size=buf_size).batch(batch_size).prefetch(1)#.make_initializable_iterator()  
    parsed_image_dataset=tf.compat.v1.data.make_initializable_iterator(parsed_image_dataset)  
    image, label = parsed_image_dataset.get_next()
    sess.run(parsed_image_dataset.initializer)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    while True:
        image= next((aug.flow(image, batch_size=bs, save_to_dir='./TrainingAugImages', save_prefix='aug', save_format='png')))
        yield image, label

def test_fn_inputs(sess, epochs, bs,  aug=None):
    train_files, total_records = get_training_data_old()
    steps_per_epoch = int(total_records / batch_size)    
    folder_path = './TestAugImages'
    shutil.rmtree(folder_path, ignore_errors = True)
    os.mkdir(folder_path)
    raw_dataset = tf.data.TFRecordDataset(train_files)     #.repeat() .repeat(epochs)
    parsed_image_dataset = raw_dataset.map(_parse_image_function).shuffle(buffer_size=buf_size).batch(batch_size).prefetch(1)#.make_initializable_iterator()  
    parsed_image_dataset=tf.compat.v1.data.make_initializable_iterator(parsed_image_dataset)  
    image, label = parsed_image_dataset.get_next()
    sess.run(parsed_image_dataset.initializer)
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    image=image.eval(session=sess)
    label=label.eval(session=sess)
    while True:
        image= next((aug.flow(image, batch_size=bs, save_to_dir='./TrainingAugImages', save_prefix='aug', save_format='png')))
        yield image, label
    """
    img_gen.append(image[0])
    label_gen.append(label[0])
    batches += 1
    if batches >= max_batches:
    break
    yield image, label
    plot_images(img_gen, label_gen, rows=2, figsize=(20,16))
    """
def plot_images2(folder, category='Aug'):
    
    batches = 0
    max_batches = 6
    img_gen = []
    label_gen = []
    label_normal_gen = []
    
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder,filename))
        if img is not None:
            #print(filename) mpimg
            img=np.array(img)
            img=Image.fromarray(img)
            img_gen.append(img)
            label_normal_gen.append(filename)
            label_gen.append('')
            batches += 1
            if batches >= max_batches:
                break

    plot_images(img_gen, label_gen, label_normal_gen, category, rows=1, figsize=(20,16))
    
    
    #return image, label

def plot_images(imgs, labels=None, rows=1, figsize=(20,8), fontsize=14):
    figure = plt.figure(figsize=figsize)
    cols = max(1,len(imgs) // rows-1)
    labels_present = False
    # checking if labels is a numpy array
    if type(labels).__module__ == np.__name__:
        labels_present=labels.any()
    elif labels:
        labels_present=True
    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols+1, i+1)
        # axis off, but leave a bounding box
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='off',
            labelleft='off')
        # plot labels if present
        if labels_present:
            subplot.set_title(labels[i], fontsize=fontsize)
        plt.imshow(imgs[i][:,:,0], cmap='Greys')
        
    plt.show()
    """
    
    sess = K.get_session()
y_true_np = sess.run(y_true) #1
y_true_np = y_true.eval(session=sess) #2
y_true_np = K.eval(y_true) #3
y_true_np = K.get_value(y_true) #4

    print(image.dtype)
    print(image.shape)
    print(image.shape[0])
    print(image[0])
    print(image[1])
    print(image[2])
    print(image.eval(session=sess))
    print(sess.run(image))
    print((np.array(image)))
    
    print(label.dtype)
    
    #image=np.array(image)
    #label=np.array(label)
    
    return next((aug.flow(image.reshape((1,) + image.shape), batch_size=bs, save_to_dir='./augmented', save_prefix='aug_', save_format='png')))
    while True:
        # if the data augmentation object is not None, apply it
        sess.run(parsed_image_dataset.initializer)
        if aug is not None:
            image, label = next(aug.flow(np.array(image, dtype=object), np.array(label, dtype=object), batch_size=bs))
        
        save_path = os.path.abspath(os.path.join(folder_path, image_data[2].decode('utf-8')))
                    mpimg.imsave(save_path, image_data[0])
        yield image, label
        #yield (np.array(image), np.array(label))
        #return parsed_image_dataset, yield (np.array(images),  to_categorical(labels,2))  .shape[0]   tf.shape
    """
def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
#tf.enable_eager_execution()
import os
import numpy as np
#import imageio
from PIL import Image
from math import floor
import pathlib
import PIL 
import IPython.display as display
import shutil
#import tensorflow.contrib.eager as tfe
import matplotlib.image as mpimg


train_path_10 = os.path.join( "..", "Dataset", "ddsm-mammography", "training10_0", "training10_0.tfrecords")
train_path_11 = os.path.join( "..", "Dataset", "ddsm-mammography", "training10_1","training10_1.tfrecords")
train_path_12 = os.path.join( "..", "Dataset", "ddsm-mammography", "training10_2", "training10_2.tfrecords")
train_path_13 = os.path.join( "..", "Dataset",  "ddsm-mammography", "training10_3", "training10_3.tfrecords")
train_path_14 = os.path.join( "..", "Dataset",  "ddsm-mammography", "training10_4", "training10_4.tfrecords")


train_path_10 = "/mdata/training10_0/training10_0.tfrecords"
train_path_11 = "/mdata/training10_1/training10_1.tfrecords"
train_path_12 = "/mdata/training10_2/training10_2.tfrecords"
train_path_13 = "/mdata/training10_3/training10_3.tfrecords"
train_path_14 = "/mdata/training10_4/training10_4.tfrecords"
       
WIDTH=299 
HEIGHT=299
IMG_WIDTH=299
IMG_HEIGHT=299
NUM_CLASSES = 2
batch_size=32
buf_size=10000

## Load the training data and return a list of the tfrecords file and the size of the dataset
## Multiple data sets have been created for this project, which one to be used can be set with the type argument
def get_training_data_old(what=10):
    if what == 10:
        train_files = [train_path_10, train_path_11, train_path_12, train_path_13, train_path_14]
        total_records = 55890
    else:
        raise ValueError('Invalid dataset!')

    return train_files, total_records

def get_test_data(what=10):
    test_files = [train_path_14]
    
    return [test_files], 11178
    

def _parse_image_function(tfrecord):
        
        # Create a dictionary describing the features.
        image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
        }
        
        image_feature_description2 = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        }
        # Parse the input tf.Example proto using the dictionary above.    image2 =tf.decode_raw(sample['image'], tf.uint8)
        #features=tf.io.parse_single_example(proto, image_feature_description2) #image1 = tf.image.decode_bmp(sample['image'], channels=3)
        
        sample=tf.parse_single_example(tfrecord, image_feature_description2) 
        image =tf.decode_raw(sample['image'], tf.uint8)
        #image = tf.image.decode_jpeg(sample['image'], channels=3)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        #image = tf.image.resize_bilinear(image, [3, 299, 299], align_corners=False)
        image = tf.reshape(image, [1, IMG_WIDTH, IMG_HEIGHT])
        #image = tf.image.resize_image_with_crop_or_pad(image, [3, IMG_WIDTH, IMG_HEIGHT])
        label = tf.cast(sample["label"], tf.int32)
        # The type is now uint8 but we need it to be float.
        image = tf.cast(image, tf.float32)
        #label = tf.reshape(label, [3, IMG_WIDTH, IMG_HEIGHT])
        label_normal = tf.cast(sample["label_normal"], tf.int32) 
        #label_normal = tf.reshape(label_normal, [3, IMG_WIDTH, IMG_HEIGHT])
        filename=2;
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label#, label_normal]  #, filename 

def validation_fn_inputs(sess, epochs, bs,  aug=None):
    aug=None    
    test_files, total_records = get_test_data()
    steps_per_epoch = int(total_records / batch_size)
    raw_dataset = tf.data.TFRecordDataset(test_files)    #.repeat(epochs).
    parsed_image_dataset = raw_dataset.map(_parse_image_function).shuffle(buffer_size=buf_size).batch(batch_size).make_initializable_iterator() 
    
    image, label = parsed_image_dataset.get_next()
    
    image = tf.reshape(image, [1, IMG_WIDTH, IMG_HEIGHT])
    #label = tf.reshape(label, [bs, 75, 25])
    
    
    while True:
        sess.run(parsed_image_dataset.initializer)
        if aug is not None:
            (images, labels) = next(aug.flow(np.array(image), np.array(label), batch_size=bs))
    	    
        yield image, label
        #yield (np.array(image), np.array(label))
    #return parsed_image_dataset, steps_per_epoch


def train_fn_inputs(sess, epochs, bs,  aug=None):
    aug=None
    train_files, total_records = get_training_data_old()
    steps_per_epoch = int(total_records / batch_size)    
    # Create folder to store extracted images
    folder_path = './ExtractedImages'
    shutil.rmtree(folder_path, ignore_errors = True)
    os.mkdir(folder_path)
    
    raw_dataset = tf.data.TFRecordDataset(train_files)     #.repeat() .repeat(epochs)
    parsed_image_dataset = raw_dataset.map(_parse_image_function).shuffle(buffer_size=buf_size).batch(batch_size).make_initializable_iterator()   
    #parsed_image_dataset = tf.compat.v1.data.make_initializable_iterator(raw_dataset.map(_parse_image_function).shuffle(buffer_size=buf_size).batch(batch_size))
    
    # session.run(parsed_image_dataset.initializer)
    image, label = parsed_image_dataset.get_next()
    
    image = tf.reshape(image, [bs, 1, IMG_WIDTH, IMG_HEIGHT])
    #label = tf.reshape(label, [bs, 75, 25])
    
    while True:
        sess.run(parsed_image_dataset.initializer)
        if aug is not None:
            (images, labels) = next(aug.flow(image, label, batch_size=bs))
        
        yield image, label
  

    iterator = parsed_image_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                image_features = sess.run(next_element)
                save_path = os.path.abspath(os.path.join(folder_path, image_features[3].decode('utf-8')))
                mpimg.imsave(save_path, image_features[0])
                #image_raw = image_features['image'].numpy()
                #display.display(display.Image(data=image_raw)) 
                
        except:
            pass
            

def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')

show(image, label)


for image, label in images_ds.take(2):
  show(image, label)

"""    

        
"""    
    for raw_record in parsed_image_dataset.take(10):
        save_path = os.path.abspath(os.path.join(folder_path, raw_record[3].decode('utf-8')))
        mpimg.imsave(save_path, raw_record[0])
    
    
    
    
    image = tf.image.decode_image(parsed_image_dataset['image'])        
    #img_shape = tf.stack([parsed_image_dataset['rows'], parsed_image_dataset['cols'], parsed_image_dataset['channels']])
    label = parsed_image_dataset['label']
    label_normal = parsed_image_dataset['label_normal']
    [image, label, label_normal]        

    
    
    iterator = parsed_image_dataset.make_one_shot_iterator().take(5)
    next_element = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                image_features = sess.run(next_element)
                save_path = os.path.abspath(os.path.join(folder_path, image_features[2].decode('utf-8')))
                mpimg.imsave(save_path, image_features[0])
                #image_raw = image_features['image'].numpy()
                #display.display(display.Image(data=image_raw)) 
                
        except:
            pass
    
    
    
    raw_dataset = raw_dataset.batch(5)
    iterator=raw_dataset.make_one_shot_iterator()
    
    for batch in iterator.get_next():
        print(batch)
        
        
    for raw_record in raw_dataset.take(10):
        print(repr(raw_record))
    image, label=read_and_decode_single_example(train_files, 'label')
    features_dataset = tf.data.Dataset.from_tensor_slices(([image, label]))
    features_dataset
    for f0,f1 in features_dataset.take(2):
            print(f0)
            print(f1)
"""

def single_image_from_disk(path_file):   
    #path_file='../Dataset/images/L-CC.png'    
    img = imageio.imread(path_file, pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def my_mammo_input_fn(filenames, labels):
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    
    """
    input_fn=This would pass the full dataset to the training code in batches of 10.
    """
    dataset_batched = dataset.batch(10)
    iterator = dataset_batched.make_one_shot_iterator()
    features, labels = iterator.get_next()
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    num_epochs=10
    batch_size=32
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    #A tf.data.Dataset that can provide data to the Keras model for training or evaluation e.g model.fit(train_dataset, epochs=3)
    return dataset

   
    
def input_fn_train():
    train_dataset_url = "https://www.kaggle.com/skooch/ddsm-mammography#training10_4.zip"
    train_label_url = "https://www.kaggle.com/skooch/ddsm-mammography#training10_4.zip"
    
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
    train_label_fp = tf.keras.utils.get_file(fname=os.path.basename(train_label_url), origin=train_label_url)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_dataset_fp, train_label_fp))
    # Shuffle and slice the dataset.
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)



def input_fn_test():
    test_dataset_url = "https://www.kaggle.com/skooch/ddsm-mammography#training10_4.zip"
    test_label_url = "https://www.kaggle.com/skooch/ddsm-mammography#training10_4.zip"
    
    test_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(test_dataset_url), origin=test_dataset_url)
    test_label_fp = tf.keras.utils.get_file(fname=os.path.basename(test_label_url), origin=test_label_url)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset_fp, test_label_fp))
    # Shuffle and slice the dataset.
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)
  
    
"""
Image loading functions
"""
def load_images_from_source2(data_dir, CLASS_NAMES):    
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    dataset=0;
    for file_path in list_ds:   #.take(5)
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, '/')
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

        # The second to last is the class-directory
        label = parts[-2] == CLASS_NAMES
        
        print("Image shape: ", img.numpy().shape)
        print("Label: ", label.numpy())
        print(file_path.numpy())
        dataset.append(img, label)
        
    return dataset


def load_images_from_source():
    data_dir = tf.keras.utils.get_file(origin='https://ddsm/ddsm.tgz', fname='mammograms', untar=True)
    #http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz
    #https://www.repository.cam.ac.uk/handle/1810/250394
    #
    #
    data_dir = pathlib.Path(data_dir)
    
    #count number of images
    #image_count = len(list(data_dir.glob('*/*.jpg')))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    load_images_from_source2(data_dir, CLASS_NAMES)
    
    #roses = list(data_dir.glob('roses/*'))
    #for image_path in roses[:3]:
        #display.display(Image.open(str(image_path)))
    return 1 


"""
Image precossing functions
"""
def my_input_preprocessing_fn():

    # Preprocess your data here...

    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return 1 #feature_cols, labels

def get_test_data2(what=10):
    test_files = os.path.join("..", "input", "ddsm-mammography", "training10_4", "training10_4.tfrecords")
    
    return [test_files], 11178

def get_training_data2(what=10):
    if what == 10:
        train_path_10 = os.path.join("..", "input", "ddsm-mammography", "training10_0", "training10_0.tfrecords")
        train_path_11 = os.path.join("..", "input", "ddsm-mammography", "training10_1","training10_1.tfrecords")
        train_path_12 = os.path.join("..", "input", "ddsm-mammography", "training10_2", "training10_2.tfrecords")
        train_path_13 = os.path.join("..", "input", "ddsm-mammography", "training10_3", "training10_3.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13]
        total_records = 44712
    else:
        raise ValueError('Invalid dataset!')

    return train_files, total_records

## read data from tfrecords file
def read_and_decode_single_example(filenames, label_type='label_normal', normalize=False, distort=False, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    reader = tf.TFRecordReader()


    if label_type != 'label':
        label_type = 'label_' + label_type

    _, serialized_example = reader.read(filename_queue)
    if label_type != 'label_mask':
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.io.FixedLenFeature([], tf.int64),
                'label_normal': tf.io.FixedLenFeature([], tf.int64),
                'image': tf.io.FixedLenFeature([], tf.string)
            })

        # extract the data
        label = features[label_type]
        image = tf.decode_raw(features['image'], tf.uint8)

        # reshape and scale the image
        image = tf.reshape(image, [299, 299, 1])

        # random flipping of image
        if distort:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

    else:
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'label': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string)
            })

        label = tf.decode_raw(features['label'], tf.uint8)
        image = tf.decode_raw(features['image'], tf.uint8)

        label = tf.cast(label, tf.int32)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        image = tf.reshape(image, [IMG_WIDTH, IMG_HEIGHT, 1])
        label = tf.reshape(label, [IMG_WIDTH, IMG_HEIGHT, 1])

        # if distort:
        #     image, label = _image_random_flip(image, label)

    if normalize:
        image = tf.image.per_image_standardization(image)

    # return the image and the label
    return image, label

def create_cbis_slices(mask_dir,image_dir,  name, debug=True):
    # initialize return variables
    image_list = []
    pixel_means_list = []
    
    # get list of files in the directory
    image_files = os.listdir(image_dir)
    counter = 0
    
    # display the progress bar
    if debug is None:
        progress(counter, len(image_files), 'WORKING')
       
    # make sure the destination directory exists
    if not os.path.exists(os.path.join("data", name)):
        os.mkdir(os.path.join("data", name))
    
    # loop through the masks
    for cur_image in image_files:
        
        # update the progress bar
        counter += 1
        if debug is None:
            progress(counter, len(image_files), cur_image)
            
        # get the image name
        base_image_file = clean_name(cur_image)
        
        full_image = PIL.Image.open(os.path.join(image_dir, cur_image))
        full_image_arr = np.array(full_image)[:,:,0]
        
        # find which masks match, there may be more than one
        matching_masks = glob.glob(os.path.join(mask_dir, base_image_file + "*" + ".jpg"))
        
        # create a blank mask same size as image
        mask_image = np.zeros_like(full_image_arr)
        
        # loop through the masks
        for mask in matching_masks:
            # load the mask
            cur_mask = np.array(PIL.Image.open(os.path.join(mask_dir, mask)))[:,:,0]
            
            # if the mask is the right shape
            if mask_image.shape == cur_mask.shape:
                # update our global mask accordingly
                mask_image[cur_mask > 0] = 1
        
        # try to remove some of the black background from the image
        mostly_black_cols = np.sum(full_image_arr < 10, axis=0)
        image_mostly_black = mostly_black_cols > (full_image_arr.shape[0] * 0.90)
        
        # determine which way the image is oriented
        first_black_col = np.argmax(image_mostly_black)
        
        # if there are a substantial number of mostly black columns then we will try to trim them
        if np.sum(image_mostly_black) > (full_image_arr.shape[1] * 0.15):
            # if the first black col is not at the beginning we trim from the right
            if first_black_col > 500:
                # add some padding
                first_black_col += 300
                
                # make sure the image is at least 1280 pixels wide
                first_black_col = np.max([first_black_col, 1280])
                
                # include up to the first black col
                full_image_arr = full_image_arr[:,:first_black_col]
                mask_image = mask_image[:,:first_black_col]

            # else we need to reverse the array and trim from the left
            else:
                first_black_col = len(image_mostly_black) - np.argmax(np.flip(image_mostly_black, axis=0))

                # add some padding
                first_black_col -= 300
                
                # make sure the image is at least 1280 pixels wide
                first_black_col = np.min([first_black_col, full_image_arr.shape[1] - 1280])
                
                # trim from the right
                full_image_arr = full_image_arr[:,first_black_col:]
                mask_image = mask_image[:,first_black_col:]
            
        # set white pixels to 1 instead of 255
#         mask_image[mask_image > 0] = 1
        
        # alert if image is way too small
        if full_image_arr.shape[1] < 1000:
            print("Image too small!", cur_image)
            print(mask, full_image_arr.shape)
        
        # make sure the mask and image are the same size
        if full_image_arr.shape != mask_image.shape:
            print("Shapes don't match", cur_image)
            continue
        else:
            # add the mean of the mask to the mean list
            pixel_means_list.append(np.mean(mask_image))
            
            # save the image
            image = np.dstack((full_image_arr, mask_image, np.zeros_like(full_image_arr))).astype(np.uint8)
            
            im = PIL.Image.fromarray(image)
            im.save(os.path.join("data", name, base_image_file + "_" + str(counter) + ".png"))
            
#         if counter > 10:
#             break
        
    # return the data
    return pixel_means_list
"""
Summary
This dataset consists of images from the DDSM [1] and CBIS-DDSM [3] datasets. The images have been pre-processed and 
converted to 299x299 images by extracting the ROIs. The data is stored as tfrecords files for TensorFlow.
The dataset contains 55,890 training examples, of which 14% are positive and the remaining 86% negative, divided into 
5 tfrecords files.
Note - The data has been separated into training and test as per the division in the CBIS-DDSM dataset. The test files 
have been divided equally into test and validation data. However the split between test and validation data was done 
incorrectly, resulted in the test numpy files containing only masses and the validation files containing only calcifications. 
These files should be combined in order to have balanced and complete test data.

Pre-processing
The dataset consists of negative images from the DDSM dataset and positive images from the CBIS-DDSM dataset. The data was 
pre-processed to convert it into 299x299 images.
The negative (DDSM) images were tiled into 598x598 tiles, which were then resized to 299x299.
The positive (CBIS-DDSM) images had their ROIs extracted using the masks with a small amount of padding to provide context.
Each ROI was then randomly cropped three times into 598x598 images, with random flips and rotations, and then the images 
were resized down to 299x299.
The images are labeled with two labels:
    label_normal - 0 for negative and 1 for positive
    label - full multi-class labels, 0 is negative, 1 is benign calcification, 2 is benign mass, 3 is malignant 
    calcification, 4 is malignant mass
The following Python code will decode the training examples:
   features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_normal': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string)
        })
    # extract the data
    label = features['label_normal']
    image = tf.decode_raw(features['image'], tf.uint8)
    # reshape and scale the image
    image = tf.reshape(image, [299, 299, 1])
The training examples do include images which contain content other than breast tissue, such as black background and 
occasionally overlay text. 
"""