import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from toolbox import newton, Dof
import csv, os 

class APE:
    
    def __init__(self,feature_matrix,y):
        self.num_training_points = y.shape[0]
        self.dim_input = feature_matrix.shape[1]
    
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
        for i in range(feature_train.shape[0]):
            if np.abs(y_train[i]) > 2.0:
                print('here')
                feature_train = np.delete(feature_train,i,axis=0)
                continue
            y_final = np.append(y_final,y_train[i]) 
        if np.abs(np.size(y_final) - np.size(y_train))/np.size(y_train) >= 0.05: 
            print('ERROR: MORE THAN 5% OF INPUT DATA 2 STDEV AWAY FROM AVERAGE')  
        return feature_train,y_final,x_bar,x_std,y_bar,y_std
    
    def train(epochs,model_type,grid,hidden_activation,output_activation,batch_size=100,loud=0,loss_tolerance=0.0009):
        if model_type == 'memory':
            weight_file = 'bananas/memory_best_weights.hdf5'
            struct_file = 'bananas/current_memory_model_structure.hdf5'
        elif model_type == 'time':
            weight_file = 'bananas/time_best_weights.hdf5'
            struct_file = 'bananas/current_time_model_structure.hdf5'
        try:
            os.remove(weight_file)
        except OSError:
            pass
        #define the epsilon-insensitive loss function
        def e_insensitive(tolerance):
            def loss(y_true, y_pred):
                return_value = K.maximum(0.0,tf.subtract(K.abs(y_true - y_pred),tolerance))
                return return_value
            return loss
        checkpoint = keras.callbacks.ModelCheckpoint(weight_file,monitor='loss',save_best_only=True)
        model = keras.models.Sequential()
        sgd_optimizer = keras.optimizers.Adam(lr=0.001)
        model.add(keras.layers.Dense(units=grid[1],input_dim=self.dim_input,kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=hidden_activation))
        for i in range(grid.shape-3):
            model.add(keras.layers.Dense(units=grid[i+2],input_dim=units,kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=hidden_activation))
        model.add(keras.layers.Dense(units=1,input_dim=units,kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=output_activation))
        model.compile(optimizer=sgd_optimizer,loss=e_insensitive(loss_tolerance))
        training_results = model.fit(feature_matrix,y,batch_size=batch_size,verbose=loud,callbacks=[checkpoint],validation_split=0.0,epochs=epochs)
        file = open(struct_file,'w')
        file.write('Units in hidden layers:'+str(units)+'\n')
        file.write('Total number of layers:'+str(layers+3)+'\n')
        file.write('Number of hidden layers:'+str(layers+1)+'\n')
        file.write('Input Layer = '+str(feature_matrix.shape[1])+' nodes\n')
        file.write('L2: input = '+str(feature_matrix.shape[1])+' output = '+str(units)+'\n')
        for i in range(layers):
            file.write('L'+str(i+3)+': input = '+str(units)+' output = '+str(units)+'\n')
        file.write('L'+str(layers+3)+': input = '+str(units)+' output = 1\n')
        file.close()
        return training_results
    
    def predict(model_type,hidden_activation,output_activation,loss_tolerance=0.0009):
        if model_type == 'memory':
            weight_file = 'bananas/memory_best_weights.hdf5'
            struct_file = 'bananas/current_memory_model_structure.hdf5'
        elif model_type == 'time':
            weight_file = 'bananas/time_best_weights.hdf5'
            struct_file = 'bananas/current_time_model_structure.hdf5'
        def e_insensitive(tolerance):
            def loss(y_true, y_pred):
                return_value = K.maximum(0.0,tf.subtract(K.abs(y_true - y_pred),tolerance))
                return return_value
            return loss 
        file = open(struct_file,'r')
        token = file.readlines()
        units = int(token[0][23:len(token[0])-1])
        layers = int(token[2][24:len(token[2])-1])-1
        print(layers)
        model = keras.models.Sequential()
        sgd_optimizer = keras.optimizers.Adam(lr=0.001)
        model.add(keras.layers.Dense(units=grid[1],input_dim=self.dim_input,kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=hidden_activation))
        for i in range(grid.shape-3):
            model.add(keras.layers.Dense(units=units,input_dim=units,kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=hidden_activation))
        model.add(keras.layers.Dense(units=1,input_dim=units,kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=output_activation))
        model.load_weights(weight_file)
        model.compile(optimizer=sgd_optimizer,loss=e_insensitive(loss_tolerance))
        output_activations = model.predict(feature_matrix)
        return output_activations
    
    def arch_optimizer():
        #define an upperbound function and its derivative
        f = lambda l: self.dim_input**l - self.num_training_points
        dfdl = lambda l: l*dim_input**(l-1)
        #use newton to find number of layers that minimizes f
        l = newton(f,dfdl,1)
        #round down becuase f is an upper bound for the size of the training set
        l = np.floor(l)
        for layers in range(l,2,-1):
            grid = np.array([])
            grid[0] = self.dim_input
            index = 1
            for i in range(grid.shape[0]-2,0,-1):
                grid[i] = self.dim_input - index
                index += 1
            grid[-1] = 1
            Dof = Dof(grid)
            iter_layer = grid[1]
            while iter_layer >= 2:
                #val_loss = network_trainer(grid,activation_info,num_epochs,model_type,batch_type,loss_tolerance)[alpha]
                df2 = pd.dataframe([grid,np.min(val_loss)],columns=['grid struct','val loss'])
                df.append(df2,ignore_index=True)
                iter_layer = grid[1] - 1
                assert Dof*self.dim_input <= self.num_training_points
        return df