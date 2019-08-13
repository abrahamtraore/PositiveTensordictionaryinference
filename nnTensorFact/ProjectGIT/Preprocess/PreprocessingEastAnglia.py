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

import pdb

import os

from scipy import signal

from scipy.io import wavfile

from melspec import melspectrogram

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
        #print(adress)
        #print(label)
        res[8*i:8*i+8,:]=result
        for l in label:
            labels.append(l)
    print(res.shape)
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

def TFR_all_the_examples_regular(res,sampling_rate,length,hop_leng):
   nbexamples=np.array(np.shape(res),dtype=int)[1]
   data=res[:,0]
   f,t,Sxx=signal.spectrogram(data,fs=sampling_rate,nperseg=length,noverlap=hop_leng)
   row=f.size
   column=t.size
   RawsampleTFR=np.zeros((nbexamples,row,column))  
   RawsampleTFR[0,:,:]=Sxx
   for i in range(nbexamples-1):
     data=res[:,i+1]
     f,t,Sxx=signal.spectrogram(data,fs=sampling_rate,nperseg=length,noverlap=hop_leng)
     RawsampleTFR[i+1,:,:]=Sxx 
   return RawsampleTFR

def TFR_all_the_examples_melscale(res,sampling_rate,length,hop_leng):   
   nbexamples=np.array(np.shape(res),dtype=int)[1]
   data=res[:,0]  
   Sxx=melspectrogram(y=data,sr=sampling_rate,n_fft=length,hop_length=hop_leng)
   sizes=np.array(np.shape(Sxx),dtype=int)
   row=sizes[0]
   column=sizes[1]
   RawsampleTFR=np.zeros((nbexamples,row,column))  
   RawsampleTFR[0,:,:]=Sxx
   for i in range(nbexamples-1):
     data=res[:,i+1]
     Sxx=melspectrogram(y=data,sr=sampling_rate,n_fft=length,hop_length=hop_leng)
     RawsampleTFR[i+1,:,:]=Sxx 
   return RawsampleTFR

def TFR_all_examples(res,sampling_rate,length,hop_leng,grid_form):
    if(grid_form=="regular"):
      RawsampleTFR=TFR_all_the_examples_regular(res,sampling_rate,length,hop_leng)
      return RawsampleTFR
    if(grid_form=="logscale"):
      RawsampleTFR=TFR_all_the_examples_melscale(res,sampling_rate,length,hop_leng)
      return RawsampleTFR
  
def Local_processing_data(filename,BASE_PATH,sampling_rate,window_length,overlap_points,grid_form,rescal):#,If=30,It=30):
    #BASE_PATH='/Users/Traoreabraham/Desktop/ProjectGit/Code/RawData'
    #filename='/Users/Traoreabraham/Desktop/ProjectGit/Data/EastAngliaDataSet'
    Datafilename=os.listdir(filename) #corresponds to the data file names
    Data=split_all_the_data_and_stack(Datafilename,filename)[0]
    print(Data)
    #sampling_rate=22050
    #window_length=2000
    #overlap_points=200
    grid_form="regular"#The other possibility is logscale
    X=TFR_all_examples(Data,sampling_rate,window_length,overlap_points,grid_form)
    print(rescal)
    print(type(rescal))
    if (rescal==1):
        print("Point A")
        If=input("Specify the width of the output images:")
        It=input("Specify the heigth of the output images:")
        result=rescaling(X,int(If),int(It))
        print(result.shape)
        K=np.array(X.shape,dtype=int)[0]
        for k in range(K):
          name="CI_"+str(k)
          file_name = name.format(k)
          np.savez(os.path.join(BASE_PATH, file_name),np.array(result[k,:,:]))
    if (rescal==0):
        K=np.array(X.shape,dtype=int)[0]
        print(X.shape)
        for k in range(K):
          name="CI_"+str(k)
          file_name = name.format(k)
          np.savez(os.path.join(BASE_PATH, file_name),np.array(X[k,:,:]))
#print("Generation EastAnglia Data")
#grid_form=input("On what kind of grid do you want to compute the signal power?:regular or logscale")
#window_length=input("Give the length of FFT window:")
#overlap_points=input("Give the number ofoverlapping points:")
#rescal=input("Do you want to resize the TFR?:the answer should be True or False")
#If=input("Give the width of the images:")
#It=input("Give the length of the images:")
#BASE_PATH='/Users/Traoreabraham/Desktop/AutomationForEastAnglia/RawData'
#filename='/Users/Traoreabraham/Desktop/AutomationForEastAnglia/Data/EastAngliaDataSet'
#sampling_rate=22050
#Local_processing_data(filename,BASE_PATH,sampling_rate,int(window_length),int(overlap_points),grid_form,str(rescal),int(If),int(It))

BASE_PATH='/Users/Traoreabraham/Desktop/AutomationForEastAnglia/RawData'
filename='/Users/Traoreabraham/Desktop/AutomationForEastAnglia/Data/EastAngliaDataSet'
sampling_rate=22050
window_length=200
overlap_points=5
grid_form=input("The form of the spectrogram:regular or logscale?")
rescal=input("Do you want to resize the TFR?:the answer should be 0(No) or 1(Yes)")
Local_processing_data(filename,BASE_PATH,sampling_rate,window_length,overlap_points,grid_form,int(rescal))#,If=50,It=50)

pdb.set_trace()
 