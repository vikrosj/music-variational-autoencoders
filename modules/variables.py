import os.path
import os.path
import numpy as np  

class HDVariables:   
    def __init__(self, path_to_variables = 'variables'):
        self.__dict__['path_to_variables'] = path_to_variables
        if not os.path.exists(self.__dict__['path_to_variables']):
            os.makedirs(self.__dict__['path_to_variables'])
            
    def __setattr__(self, name, value):
        np.save(self.__dict__['path_to_variables'] + '/' + name +'.npy', value)
        self.__dict__[name] = value
        
    def __getattr__(self, name):
        if (os.path.isfile(self.__dict__['path_to_variables'] + '/' + name +'.npy')):
            return np.load(self.__dict__['path_to_variables'] + '/' + name +'.npy')
        else:
            return None