import argparse
from scgal import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scgal")
    parser.add_argument('--epochs', type=int, default=200, help='number of epoches to train the scgal.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.006, help='learning rate.')
    parser.add_argument('--Kmax', type=int, default=30, help='the maximum number of clusters to consider.')
    parser.add_argument('--latent_dim', type=int, default=3, help='the latent dimension.')
    parser.add_argument('--A-col', default='1', help='scDNA-seq data dimensions')
    parser.add_argument('--B-col', default='1', help='scRNA-seq data dimensions')
    parser.add_argument('--lambda_A', type=float, default=5, help='a weight factor to balance reconstruction loss and adversarial loss')
    parser.add_argument('--gan_mode', type=str, default='vanilla',help='vanilla GAN loss is used in the original GAN paper.')
    parser.add_argument('--dna', type=str, default='./data/NCI-N87/N87_dna.csv')
    parser.add_argument('--rna', type=str, default='./data/NCI-N87/N87_rna.txt')
    parser.add_argument('--is_true', type=int, default=False)
    parser.add_argument('--pool_size', type=int, default=64,help='the size of rna buffer that stores previously generated rnas')
    parser.add_argument('--seed', type=int, default=0,help='random seed.')
    opt = parser.parse_args()
    main(opt)