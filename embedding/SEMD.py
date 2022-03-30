# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 18:44:19 2022

@author: DELL
"""

from sklearn.metrics import r2_score
import numpy as np
import scipy.io as scio
import os 
import csv
from VAE3 import *
import time
startTime = time.time()
def feature_normalization(feat_norm):
    for i in range(0,feat_norm.shape[1]):
        norm_temp = np.linalg.norm(feat_norm[:,i],ord=2)
        if norm_temp > 0:
             feat_norm[:,i] = feat_norm[:,i]/norm_temp;
    return feat_norm

def matrix_hstack(feat):
    feat_feat = feat[0]
    for i in range(1,len(feat)):
        feat_feat_temp = feat[i]
        feat_feat =  np.hstack((feat_feat, feat_feat_temp))
        feat_feat = feature_normalization(feat_feat)
    return feat_feat 

def feiling(interaction,dd):
    aa = []
    bb = []
    ii=  0
    sum = 0
    for i in range(np.shape(interaction)[0]): # 行732
        for j in range(np.shape(interaction)[1]): # 列 1915
            if interaction[i][j]!=0:
                ii=ii+1
                temp = np.square(interaction[i][j]-dd[i][j])
                sum = sum + temp 
                aa.append(interaction[i][j])
                bb.append(dd[i][j])
   # print(sum)
   # value =np.sqrt(sum/ii)
    value =r2_score(aa,bb)
    print(value)
    return value

def outputCSVfile(filename,data):
    csvfile=open(filename,'w')
    writer=csv.writer(csvfile)
    writer.writerows(data)
    csvfile.close()

dataFile = r'C:\Users\DELL\Desktop\DTI\dti\code\feature'
file = os.listdir(dataFile)
drugFeature = []
proteinFeature = []

for file_index  in  file:
    data = scio.loadmat(dataFile + '//' + file_index)
    print(file_index)
    data = data['rep_sim1_drug']
    data = data.astype('float32') / np.max(data)
    original_dim =  data.shape[0]
    if original_dim == 732:
      #  drugF = FeatureExtraction(original_dim,data)
        drugFeature.append(data) 
    else:
       # proteinF = FeatureExtraction(original_dim,data)
        proteinFeature.append(data) 
        
        
drugF = matrix_hstack(drugFeature)      
drug_feat, decoded_drug = FeatureExtraction_d(drugF.shape[1],drugF)
N_drugF = matrix_hstack(decoded_drug)      

drug =[]
N_d =  int(np.shape(drugF)[1]/9)
for i in (range(0,9)): # 行732
        ii = i*N_d
        temp =  feiling(drugF[:,ii:ii+N_d],N_drugF[:,ii:ii+N_d])
        drug.append(temp)
        
proteinF = matrix_hstack(proteinFeature)  
prot_feat,decoded_protein = FeatureExtraction_p(proteinF.shape[1],proteinF)        
N_protienF = matrix_hstack(decoded_protein)  


protien =[]
N_t =  int(np.shape(proteinF)[1]/6)
for j in (range(0,6)): # 行732
        jj = j*N_t
        temp =  feiling(proteinF[:,jj:jj+N_t],N_protienF [:,jj:jj+N_t])
        protien.append(temp)
#scio.savemat('drug_feat.mat', {'drug_feat':drug_feat})
#scio.savemat('prot_feat.mat', {'prot_feat':prot_feat})



outputCSVfile('drugFeature.txt',drug_feat)
outputCSVfile('proteinFeature.txt',prot_feat)
# endTime = time.time()
# tt  = endTime-startTime
# print('The time of code  is: %s' %tt)