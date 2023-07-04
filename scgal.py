import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import torch
import copy
import numpy as np
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score
from sklearn.metrics import silhouette_score
from scgal_model import Create_Model,set_requires_grad,GANLoss
from rna_pool import Pool
from clustering import GMM
from data.data_process import xs_gen
from data import create_dataset
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Used to specify which device to use to run the PyTorch model

criterionGAN = GANLoss('vanilla').to(device)  # define GAN loss.
ceiteriondAE = torch.nn.MSELoss()             # define AE loss.
criterionD = GANLoss('vanilla').to(device)    # define D loss

def main(opt):

    start_t = datetime.datetime.now()     # Get current time
    optimizers = []
    data = create_dataset(opt)            # Create dataset
    data_bk = copy.deepcopy(data)
    data_A_size = data.dataset.A_data.shape[0]
    opt.A_col = data.dataset.A_data.shape[1]
    opt.B_col = data.dataset.B_data.shape[1]
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    setup_seed(opt.seed)

    model,modelD = Create_Model(opt) # Create Model
    model = model.to(device)
    modelD = modelD.to(device)

    optimizer_ae = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(modelD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    optimizers.append(optimizer_ae)
    optimizers.append(optimizer_D)

    epochs = []
    train_loss_AE = []
    train_loss_GAN = []
    train_loss_D = []
    fake_B_pool = Pool(opt.pool_size)  # create rna buffer to store previously generated rnas

    # Start training the model
    model.train()
    modelD.train()
    print("Iteration Start:")
    for epoch in tqdm(range(opt.epochs)): # Iteration Start
        epoch_iter = 0           # Record the number of iterations of this epoch and reset to 0 in each epoch
        data_train = copy.deepcopy(data)
        for step, x in xs_gen(data_train, opt.batch_size, 1):
            epoch_iter += opt.batch_size
            if epoch_iter<=data_A_size:
                vis = float(epoch_iter) / data_A_size +epoch
                epochs.append(vis)
            real_A = x['A'].to(device)
            real_B = x['B'].to(device)
            h_enc,rec,fake= model(real_A)

            set_requires_grad(modelD, False)  # modelD requires no gradients when optimizing model
            optimizer_ae.zero_grad()  # set model's gradients to zero
            lambda_A = opt.lambda_A

            loss_decoder1 = criterionGAN(modelD(fake), True)
            loss_decoder2 = ceiteriondAE(rec,real_A)* lambda_A

            # combined loss and calculate gradients
            loss_cycle_ae = loss_decoder1 + loss_decoder2
            loss_cycle_ae.backward()   # calculate gradients for model
            optimizer_ae.step()  # updata model's weights
            # D_A and D_B
            set_requires_grad(modelD, True)
            optimizer_D.zero_grad()  #  set modelD's gradients to zero
            # Real
            pred_real = modelD(real_B.detach())
            loss_D_real = criterionD(pred_real, True)
            # Fake
            RNA_fake_pool = fake_B_pool.query(fake)
            pred_fake = modelD(RNA_fake_pool.detach())
            loss_D_fake = criterionD(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()   # calculate gradients for modelD
            optimizer_D.step()  # updata modelD's weights
            if epoch_iter <= data_A_size:
                train_loss_AE.append(loss_decoder2.data.cpu().numpy())
                train_loss_GAN.append(loss_decoder1.data.cpu().numpy())
                train_loss_D.append(loss_D.data.cpu().numpy())

    # get latent representation of single cells after VAE training is completed
    a = []
    data_eval = copy.deepcopy(data_bk)
    model.eval()
    modelD.eval()
    for step, x in xs_gen(data_eval, opt.batch_size, 0):
        real_A_eval  = x['A'].to(device)
        with torch.no_grad():
            h_enc,rec,fake = model(real_A_eval)
            z=h_enc.cpu().detach().numpy()
            a.append(z)
    for id, mu in enumerate(a):
        if id == 0:
            features = mu
        else:
            features = np.r_[features, mu]

    features_array = np.array(features)
    np.savetxt('./results/latent.txt', features_array,fmt='%.6f', delimiter=',') # Save features to TXT file
    print("latent.txt saved successfully")
    # use Gaussian mixture model to cluster the single cells
    print('clustering the cells...')

    if opt.Kmax <= 0:
        kmax = np.max([1, features.shape[0] // 10])
    else:
        kmax = np.min([opt.Kmax, features.shape[0] // 10])
    print("GAN_AE:")
    label_p, K = GMM(features, kmax).cluster()
    label_p_array = np.array(label_p)
    label_p_array = label_p_array.reshape(1, -1)
    np.savetxt('./results/label.txt', label_p_array, fmt='%d',delimiter=',')  #Save label_p to TXT file
    print("label.txt saved successfully")
    print('inferred number of clusters: {}'.format(K))
    if opt.is_true:
        label_t = np.loadtxt(opt.label, dtype='float32', delimiter=',')
        v_measure = v_measure_score(label_t, label_p)   # With real label, calculate v_measure and ARI
        ari = adjusted_rand_score(label_t, label_p)
        print('V-measure: ', v_measure, 'ARI: ', ari)
    else:
        if K==1:
            sc_score = 0.0
        else:
            sc_score = silhouette_score(features, label_p)    # No real label,calculate silhouette coefficient
        print('silhouette_score: ', sc_score)

    end_t = datetime.datetime.now()
    print('elapsed time: ', (end_t - start_t).seconds)



