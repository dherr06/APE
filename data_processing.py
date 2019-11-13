import numpy as np
import csv

class data_processing:
    
    def input_generator(filepath):
        y_final = np.array([])
        with open(filepath,'r') as readfile:
            reader = csv.reader(readfile)
            index = 0
            run_size = 0
            for row in reader:
                if index > 0 and row[0][0] != 'N':
                    run_size += 1
                    row = np.array(row,dtype='float')
                    cols = np.size(row)
                index += 1
        feature_matrix = np.zeros((run_size,cols))
        with open(filepath,'r') as readfile:
            reader = csv.reader(readfile)
            index = 0
            for row in reader:
                if index > 0 and row[0][0] != 'N':
                    row = np.array(row,dtype='float')
                    for feature in range(np.size(row)):
                        feature_matrix[index-2][feature] = row[feature]
                index += 1
        #split the y values from the feature matrix
        y_train = feature_matrix[:,feature_matrix.shape[1]-1]
        feature_matrix = np.delete(feature_matrix,feature_matrix.shape[1]-1,axis=1)
        #remove overall runtime, grind time, and sweep time
        feature_matrix = np.delete(feature_matrix,feature_matrix.shape[1]-1,axis=1)
        feature_matrix = np.delete(feature_matrix,feature_matrix.shape[1]-1,axis=1)
        feature_matrix = np.delete(feature_matrix,feature_matrix.shape[1]-1,axis=1)
        #remove irrelevant data
        x_std = np.std(feature_matrix,axis=0)
        keep_vec = []
        for i in range(np.size(x_std)):
            if x_std[i] != 0:
                keep_vec = np.append(keep_vec,i)
        feature_train = np.zeros((feature_matrix.shape[0],np.size(keep_vec)))
        for i in range(np.size(keep_vec)):
            feature_train[:,i] = feature_matrix[:,int(keep_vec[i])]
        #normalize training data
        x_std = np.std(feature_train,axis=0)
        x_bar = np.mean(feature_train,axis=0)
        y_std = np.std(y_train)
        y_bar = np.mean(y_train)
        assert np.size(y_train) == feature_train.shape[0]
        feature_train = (feature_train - x_bar)/x_std
        y_train = (y_train - y_bar)/y_std
        #check for data points greater than 2 stdev's away from average
        delete_vec = np.array([])
        for i in range(feature_train.shape[0]):
            if np.abs(y_train[i]) > 3.0:
                delete_vec = np.append(delete_vec,i)
                continue
            y_final = np.append(y_final,y_train[i])
        feature_train = np.delete(feature_train,delete_vec,axis=0)
        assert feature_train.shape[0] == y_final.shape[0]
        if np.abs(np.size(y_final) - np.size(y_train))/np.size(y_train) >= 0.05: 
            print('ERROR: MORE THAN 5% OF INPUT DATA 2 STDEV AWAY FROM AVERAGE')  
        return feature_train,y_final,x_bar,x_std,y_bar,y_std