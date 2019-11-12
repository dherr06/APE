import numpy as np
import csv, math
from APE import APE

def main():
    #initialize some containers
    y_memory = np.array([])
    y_time = np.array([])
    delete_vec = np.array([])
    #build feature matrix and fill RHS vectors
    with open('run_data_P_1.csv','r') as readfile:
        reader = csv.reader(readfile)
        counter = 0
        for row in reader:
            if counter > 1: 
                row = np.array(row,dtype='float')
                y_time = np.append(y_time,row[np.size(row)-2])
                y_memory = np.append(y_memory,row[np.size(row)-1])
            counter += 1
    feature_matrix_time = np.zeros((counter-2,15))
    with open('run_data_P_1.csv','r') as readfile:
        reader = csv.reader(readfile)
        counter = 0
        for row in reader:
            if counter > 1:
                row = np.array(row,dtype='float')
                for i in range(15):
                    feature_matrix_time[counter-2][i] = row[i]
            counter += 1
    #normalize data
    x_std = np.std(feature_matrix_time,axis=0)
    x_bar = np.mean(feature_matrix_time,axis=0)
    y_memory_std = np.std(y_memory)
    y_memory_bar = np.mean(y_memory)
    y_time_std = np.std(y_time)
    y_time_bar = np.mean(y_time)
    feature_matrix_time = (feature_matrix_time - x_bar)/x_std
    y_memory = (y_memory - y_memory_bar)/y_memory_std
    y_time = (y_time - y_time_bar)/y_time_std
    #delete columns from feature matrix that have zero std
    for row in feature_matrix_time:
        for element in range(np.size(row)):
            if math.isnan(row[element]):
                delete_vec = np.append(delete_vec,element)
    feature_matrix_time = np.delete(feature_matrix_time,delete_vec,axis=1)
    delete_vec = np.array([])
    #clone the feature matrix so we can have two seperate data sets for memory and time
    #there are potentially a significant amount of data points that are outliers for one 
    #data set but not the other
    feature_matrix_memory = feature_matrix_time.copy()
    #delete outliers from both data sets
    for i in range(np.size(y_memory)):
        if np.abs(y_memory[i]) > 2:
            delete_vec = np.append(delete_vec,i)
    y_memory = np.delete(y_memory,delete_vec,None)
    feature_matrix_memory = np.delete(feature_matrix_memory,delete_vec,axis=0)
    delete_vec = np.array([])
    for i in range(np.size(y_time)):
        if np.abs(y_time[i]) > 2:
            delete_vec = np.append(delete_vec,i)
    y_time = np.delete(y_time,delete_vec,None)
    feature_matrix_time = np.delete(feature_matrix_time,delete_vec,axis=0)
    delete_vec = np.array([])
    #train the networks
    sigmoid = lambda x:1/(1+np.exp(-x))
    time_history = APE.train(sigmoid(feature_matrix_time),sigmoid(y_time),50000,'time',layers=3,loss_tolerance=0.001,loud=0,batch_size=200,units=40)
    memory_history = APE.train(sigmoid(feature_matrix_memory),sigmoid(y_memory),50000,'memory',layers=2,loss_tolerance=0.001,loud=0,batch_size=200,units=40)
    
if __name__ == '__main__':
    main() 
