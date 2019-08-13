#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:48:05 2017

@author: Traoreabraham
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:23:58 2017

@author: Traoreabraham
"""
import numpy as np

from scipy.io import wavfile

from PIL import Image 

from scipy import misc

def split_one_data_into_chunks_of_8_signals(data,label_number):
    #This function is used to cut each single signal into eight signals
    label=label_number*np.ones(8)
    size_of_chunks=661500   
    result=np.zeros((8,size_of_chunks))   
    for k in range(8):
        result[k,:]=data[k*size_of_chunks:(k+1)*size_of_chunks]
    return result,label

def split_all_the_data_and_stack(Datafilename,filename):
    #This function is used to cut each of the ten signals into eight signal(which yields the 80 samples) and recover the corresponding labels
    labels=[]
    size_of_chunks=661500
    res=np.zeros((80,size_of_chunks))
    for i in  range(len(Datafilename)):
        adress=filename+"/"
        adress=adress+Datafilename[i]
        samplerate,data=wavfile.read(adress)     
        result,label=split_one_data_into_chunks_of_8_signals(data,i+1)
        res[8*i:8*i+8,:]=result
        for l in label:
            labels.append(l) 
    labels=np.array(labels,dtype=int)
    return np.transpose(res),labels

def rescaling(image_data,width,length):
      #This function is used to reduce the dimensions of the TFR images
      size=np.array(image_data.shape,dtype=int)
      number_of_samples=size[0]
      result=np.zeros((number_of_samples,width,length))
      for k in range(number_of_samples):
          image_of_interest=image_data[k,:,:]
          image_of_interest=Image.fromarray(image_of_interest)
          image_of_interest=misc.imresize(image_of_interest,(width,length),interp='cubic')
          image_of_interest=np.array(image_of_interest)
          result[k,:,:]=image_of_interest
      return result

def Recover_classes_one_label(features,labels,label_number):
    #This function is used to recover all the samples belonging to one class 
    number_samples=np.size(labels)
    result=[]
    for i in range(number_samples):
        if(labels[i]==label_number):
            result.append(features[i])
    result=np.array(result)
    return result

def suppress_repetition_in_a_list(labels):
    labels_list=list(labels) 
    return list(set(labels_list))

def Recover_classes_all_labels(features,labels):
    #This function is used to recover all the examples with their respective labels
    list_labels=suppress_repetition_in_a_list(labels)
    number_samples=len(list_labels)
    result=[]
    class_indexes=[]
    number_of_samples_per_class=[]
    for i in range(number_samples):
        if(list_labels[i]==1):
            result.append(Recover_classes_one_label(features,labels,1))
            result.append(1)
            class_indexes.append(1)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,1))[0])
        if(list_labels[i]==2):
            result.append(Recover_classes_one_label(features,labels,2))
            result.append(2)
            class_indexes.append(2)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,2))[0])
        if(list_labels[i]==3):
            result.append(Recover_classes_one_label(features,labels,3))
            result.append(3)
            class_indexes.append(3)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,3))[0])
        if(list_labels[i]==4):
            result.append(Recover_classes_one_label(features,labels,4))
            result.append(4)
            class_indexes.append(4)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,4))[0])
        if(list_labels[i]==5):
            result.append(Recover_classes_one_label(features,labels,5))
            result.append(5)
            class_indexes.append(5)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,5))[0])
        if(list_labels[i]==6):
            result.append(Recover_classes_one_label(features,labels,6))
            result.append(6)
            class_indexes.append(6)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,6))[0])
        if(list_labels[i]==7):
            result.append(Recover_classes_one_label(features,labels,7))
            result.append(7)
            class_indexes.append(7)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,7))[0])
        if(list_labels[i]==8):
            result.append(Recover_classes_one_label(features,labels,8))
            result.append(8)
            class_indexes.append(8)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,8))[0])
        if(list_labels[i]==9):
            result.append(Recover_classes_one_label(features,labels,9))
            result.append(9)
            class_indexes.append(9)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,9))[0])
        if(list_labels[i]==10):
            result.append(Recover_classes_one_label(features,labels,10))
            result.append(10)
            class_indexes.append(10)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,10))[0])
    return result,class_indexes ,number_of_samples_per_class

