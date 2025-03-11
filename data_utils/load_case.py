'''
Currently vortex_2d_ns only does 1 reynolds number sliding window but is not compatiable with the
PDE class object.

The general version will do sliding window (period) for multiple Reynolds numbers and different
setups all compatiable with the PDE class object
'''

def load_dataset_case(config):
    if config['name'] == 'Vortex_2D':
        from data_utils.vortex_2d_ns import Cylinder2DDataset as Dataset
        train_set = Dataset(config, Re_index=-1, train=True) #145 when looking at old dataset
        val_set = Dataset(config, Re_index=-1, train=False)
        '''
        TODO: Move the Re_index into the dataset args.. or use the *args function
        '''
    if config['name'] == 'Vortex_2D_v2':
        from data_utils.vortex_2d_ns_general import Cylinder2DDataset as Dataset
        train_set = Dataset(config, train=True) #145 when looking at old dataset
        val_set = Dataset(config, train=False)
    else:
        raise NotImplementedError(f'{config["name"]} not implemented yet')
    
    return train_set, val_set