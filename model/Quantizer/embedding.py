from train_ae_unchanged import set_seed, resize_data, CONFIG  # Reusing from train script
import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from vector_quant import VectorQuantizer, VectorQuantizerEMA
from enc_dec import IMUEncoder, IMUDecoder
from scipy.interpolate import interp1d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from imu_test import Model as VQ_VAE_Model
print(device)
import os


# Set random seed
set_seed(CONFIG["random_seed"])



# Load dataset

root_path = "/home/dfki/dee/VQKANClassifier-main/Datasets/"
dataset = "wisdm_30hz_clean"
X = np.load(f"{root_path}/{dataset}/X.npy")
Y = np.load(f"{root_path}/{dataset}/Y.npy")


device = "cuda" if torch.cuda.is_available() else "cpu"
model = VQ_VAE_Model(**CONFIG["model_params"]).to(device)
model_path = CONFIG["model_save_path"]  # Change fold if needed
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
all_embeddings = []

final_codebook_batches=[]
final_embedding_batches = []
for j in range(X.shape[0]):                  #(X.shape[0]):  
    embedding_datas = []
    codebook = []
    total_time = X.shape[1]
    num_windows = total_time // CONFIG["input_length"]
    end_Col = num_windows*CONFIG["input_length"]
    num_windows+=1 if end_Col != total_time else num_windows
    label = Y[j]
    # print('Y:',label)

    for i in range(num_windows):
        if i < num_windows-1:
            input_data = X[1, i * CONFIG["input_length"]:(i + 1) * CONFIG["input_length"], :]
            input_data = np.expand_dims(input_data, axis=0)
            # print(f'input daza the layer{i} batch{j}',input_data.shape)
            
            input_data = torch.tensor(input_data, dtype=torch.float32)
            input_data = np.transpose(input_data, (0, 2, 1)) 
            with torch.no_grad():
                loss, _, perplexity,quantized_embeddings= model(input_data.to(device))
                # quantized_embeddings = quantized_embeddings.cpu().numpy()
                quantized_embeddings = quantized_embeddings.detach().cpu().numpy()
                # print(quantized_embeddings.shape)

            
            # print(model._vq_vae._embedding.weight.shape)
            # codebook.append(model._vq_vae._embedding.weight)
            # codebook.append(model._vq_vae._embedding.weight.detach().cpu())
            # codebook.append(model._vq_vae._embedding.weight.cpu().detach())



            # print(codebook)
            # print(quantized_embeddings.shape)
            embedding_datas.append(quantized_embeddings)
        else:
            input_data = X[j,end_Col:,:]
            input_data = np.expand_dims(input_data, axis=0) 
     
            dee = input_data.shape[1]
            pad_value = CONFIG["input_length"] - input_data.shape[1]
            input_data = np.pad(input_data, ((0, 0), (0, pad_value), (0, 0)), mode='constant', constant_values=0)
            # print(f'input daza the layer{i} batch{j}',input_data.shape)
            input_data = torch.tensor(input_data, dtype=torch.float32)
            input_data = np.transpose(input_data, (0, 2, 1)) 
            with torch.no_grad():
                loss, _, perplexity,quantized_embeddings= model(input_data.to(device))
                # quantized_embeddings = quantized_embeddings.cpu().numpy()
                quantized_embeddings = quantized_embeddings.detach().cpu().numpy()
                # print(quantized_embeddings.shape)

            
            # print(model._vq_vae._embedding.weight.shape)
            # codebook.append(model._vq_vae._embedding.weight)
            # codebook.append(model._vq_vae._embedding.weight.detach().cpu())
            # codebook.append(model._vq_vae._embedding.weight.cpu().detach())



            # print(codebook)
            # print(quantized_embeddings.shape)
            embedding_datas.append(quantized_embeddings)
             
            
    # stacked_codebook = torch.stack(codebook,dim=0).detach()   #if 4*1028,128 then np.cat
    # print(len(embedding_datas))
    stacked_data_time = np.concatenate(embedding_datas, axis=0)
    t1,t2,t3 = stacked_data_time.shape
    # print(stacked_data_time.shape)

    batch_dict = {"data": stacked_data_time.reshape(t1,-1), "label": label}
    # codebook_with_labels = {"data": stacked_codebook}
    save_path = "embeddings_llama_70b"
    os.makedirs(save_path, exist_ok=True)
    np.savez(f"{save_path}/embedding_batch_{j}.npz", batch_dict)
    
    final_embedding_batches.append(batch_dict)
    # final_codebook_batches.append(codebook_with_labels)
    # for i, batch in enumerate(batch_dict):
    #     print(f"Batch {i}: Data Shape = {batch['data'].shape}, Label = {batch['label']}")

    # batch_dict ={}

    # for i, batch in enumerate(batch_dict):
    #     print(f"Batch {i}: Data Shape = {batch['data'].shape}, Label = {batch['label']}")
    # Save intermediate results to free memory
    # np.save(f"embedding_batch_{j}.npy", stacked_data_time)
    # torch.save(stacked_codebook, f"codebook_batch_{j}.pt")
    # 
    del embedding_datas, codebook, stacked_data_time #, stacked_codebook
    torch.cuda.empty_cache()  # Free GPU memory

# Clear lists to avoid keeping unnecessary references


for i, batch in enumerate(final_embedding_batches):
        print(f"Batch {i}: Data Shape = {batch['data'].shape}, Label = {batch['label']}")


# final_embedding_batches.clear()
# final_codebook_batches.clear()


#     batch_dict = {
#             "data": stacked_data_time.squeeze(),  # Remove singleton dimension
#             "label": label
#         }
    
#     codebook_with_labels = {
#             "data": stacked_codebook,  
#         }
#     # print(stacked_data_time.shape)
#     # print(stacked_codebook.shape)
#     final_embedding_batches.append(batch_dict)
#     final_codebook_batches.append(codebook_with_labels)


