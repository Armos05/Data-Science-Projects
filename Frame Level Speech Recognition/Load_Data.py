import numpy as np
from pandas import read_csv

class WSJ():

    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None
    
    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw('../input/masterclass-hw-1/', 'dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw('../input/masterclass-hw-1/', 'train')
        return self.train_set
  
    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join('../input/masterclass-hw-1/', 'test_data.npy'), encoding='bytes'), None)
        return self.test_set
    
def load_raw(path, name):
    print(os.path.join(path, '{}_data.npy'.format(name)))
    return (
        np.load(os.path.join(path, '{}_data.npy'.format(name)), encoding='bytes'), 
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
        
        #np.load('dev_data.npy', encoding='bytes',allow_pickle=True), 
        #np.load('dev_labels.npy', encoding='bytes',allow_pickle=True)
    )
