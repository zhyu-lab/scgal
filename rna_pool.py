import random
import torch


class Pool():
    """
    This class implements an data buffer that stores previously generated rnas.
    This buffer enables us to update discriminators using a history of generated rnas
    rather than the ones produced by the latest generators.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_rnas = 0
            self.rnas = []
    def query(self, rnas):
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return rnas
        return_rnas = []
        for rna in rnas:
            rna = torch.unsqueeze(rna.data, 0)
            if self.num_rnas < self.pool_size:   # if the buffer is not full; keep inserting current rnas to the buffer
                self.num_rnas = self.num_rnas + 1
                self.rnas.append(rna)
                return_rnas.append(rna)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored scRNA-seq data, and insert the current scRNA-seq data into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.rnas[random_id].clone()
                    self.rnas[random_id] = rna
                    return_rnas.append(tmp)
                else:       # by another 50% chance, the buffer will return the current scRNA-seq data
                    return_rnas.append(rna)
        return_rnas = torch.cat(return_rnas, 0)   # collect all the rnas and return
        return return_rnas
