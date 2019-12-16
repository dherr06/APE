import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from toolbox import Dof
from scipy.optimize import fsolve
import csv, os 

class APE:
    
    def __init__(self,feature_matrix,y,val_x,val_y,hidden_activation,output_activation):
        self.feature_matrix = feature_matrix
        self.y = y
        self.val_x = val_x
        self.val_y = val_y
        self.num_training_points = y.shape[0]
        self.dim_input = feature_matrix.shape[1]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def network_constructor(self,grid,tolerance,compile_=True):
        model = keras.models.Sequential()
        #define the epsilon-insensitive loss function
        def e_insensitive(tolerance):
            def loss(y_true, y_pred):
                return_value = K.maximum(0.0,tf.subtract(K.abs(y_true - y_pred),tolerance))
                return return_value
            return loss
        sgd_optimizer = keras.optimizers.Adam(lr=0.001)
        model.add(keras.layers.Dense(units=grid[1],input_dim=self.dim_input,kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=self.hidden_activation))
        for i in range(grid.shape[0]-3):
            model.add(keras.layers.Dense(units=grid[i+2],input_dim=grid[i+1],kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=self.hidden_activation))
        model.add(keras.layers.Dense(units=1,input_dim=grid[-2],kernel_initializer='glorot_normal',bias_initializer=keras.initializers.Constant(0.0),activation=self.output_activation))
        if compile_:
            model.compile(optimizer=sgd_optimizer,loss=e_insensitive(tolerance))
        return model
    
    def train(self,grid,epochs,model_type,batch_size=1000,loud=0,loss_tolerance=0.001):
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
        model = APE.network_constructor(self,grid,loss_tolerance)
        es = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=2000)
        checkpoint = keras.callbacks.ModelCheckpoint(weight_file,monitor='val_loss',save_best_only=True)
        training_results = model.fit(self.feature_matrix,self.y,batch_size=batch_size,verbose=loud,callbacks=[checkpoint,es],validation_data=[self.val_x,self.val_y],epochs=epochs)
        file = open(struct_file,'w')
        for element in range(grid.shape[0]):
            file.write(str(grid[element])+'\n')
        return training_results
    
    def predict(self,feature_matrix,model_type,loss_tolerance=0.001):
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
        grid = np.array([])
        for line in token:
            grid = np.append(grid,line)
        grid = np.array(grid,dtype='float')
        model = APE.network_constructor(self,grid,loss_tolerance,compile_=False)
        sgd_optimizer = keras.optimizers.Adam(lr=0.001)
        model.load_weights(weight_file)
        model.compile(optimizer=sgd_optimizer,loss=e_insensitive(loss_tolerance))
        output_activations = model.predict(feature_matrix)
        return output_activations
    
    def arch_optimizer(self,max_layer=1,max_neurons=1):
        grid = np.array([10,1,1])
        #while we havent hit the max number of hidden layers, keep doing the analysis
        while grid.shape[0] <= max_layer+2:
            hl = grid.shape[0]-2
            history = APE.train(self,grid,10000,'time',loud=0)
            data = {'history':[history], 'grid_struct':[grid.copy()]}
            df2 = pd.DataFrame(data = data)
            df = df.append(df2,ignore_index=True)
            #train and test, then add a neuron to the next layer, then repeat
            layer = 1
            for i in range(int(hl*(max_neurons-1))):
                grid[layer] += 1
                history = APE.train(self,grid,10000,'time',loud=0)
                data = {'history':[history], 'grid_struct':[grid.copy()]}
                df2 = pd.DataFrame(data = data)
                df = df.append(df2,ignore_index=True)
                print('just ran with the above grid')
                if layer == grid.shape[0]-2:
                    layer = 1
                else:
                    layer += 1
            grid = np.append(grid,1)
            grid = np.ones(grid.shape[0])
            grid[0] = self.dim_input
        return df