import torch
import torch.utils.data as data
import numpy as np

# Directory containing the data.
root = 'data/cmb_data'

def get_celeba(params, ):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data pr
    true_dataset = Dataset(params['file_path'])
    masked_dataset = Dataset(params['file_path'], mask_file = params['mask'], mask=True)

    
    # Create the dataset.
    

    # Create the dataloader.
    true_dataloader = torch.utils.data.DataLoader(true_dataset,
        batch_size=params['bsize'],
        shuffle=False)

    masked_dataloader = torch.utils.data.DataLoader(masked_dataset,
    batch_size=params['bsize'],
    shuffle=False)


    return true_dataloader, masked_dataloader

class Dataset(data.Dataset):
    def __init__(self, file_path, mask_file=None, mask=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        super(data.Dataset, self).__init__()
        self.npmaps = np.load(file_path)
        self.mask = mask
        if mask==True:
            self.mask_vec = np.load(mask_file)

    def __len__(self):
        return len(self.npmaps)

    def __getitem__(self, idx):
        sample = self.npmaps[idx]
        if self.mask==True:
            masked_sample = sample*self.mask_vec
            masked_sample[masked_sample==0]=-1
            return torch.Tensor(masked_sample)
        return torch.Tensor(sample)
