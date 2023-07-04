import torch.utils.data
import data.data_process as data_p

def create_dataset(opt):
    """Create a dataset given the option.
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset =data_p.data_set(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            )

    def load_data(self):
        return self
