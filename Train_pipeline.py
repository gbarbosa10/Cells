from Loss import loss
from Dataset import Dataset
from Unet import UNet

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import cv2
import pandas as pd
import numpy as np
import time

import random
import cv2
from PIL import Image

learning_rate = 1e-3
batch_size = 32
epochs = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

SEPARATE_FILE_IN_TEST = True
PATH = 'C:/Users/guiti/Documents/INEGI/mariana/'
CROP_SIZE = 200
BATCH_SIZE = 32

#Read Torch Data Files

data_fil1 = torch.load(PATH + '32.3_N_11.29.18.pt')
data_fil2 = torch.load(PATH + '33.6_H_11.29.18.pt')
data_fil3 = torch.load(PATH + '33.6_N_11.29.18.pt')

file1 = np.arange(0, 10)
file2 = np.arange(10, 20)
file3 = np.arange(20, 30)

data_csv1 = pd.read_csv(PATH + '32.3_N_11.29.18_centroids.csv')
data_csv2 = pd.read_csv(PATH + '33.6_H_11.29.18_centroids.csv') 
data_csv2["img_lb"] = data_csv2["img_lb"].add(10)
data_csv3 = pd.read_csv(PATH + '33.6_N_11.29.18_centroids.csv') 
data_csv3["img_lb"] = data_csv3["img_lb"].add(20)

if (SEPARATE_FILE_IN_TEST):
    train_data = torch.cat((data_fil1, data_fil2))
    train_lb = np.concatenate([file1, file2])

    X_test = data_fil3
    y_test = data_csv3["img_lb"]

    X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_lb , test_size=0.2, random_state=0)

else:
    data = torch.cat((data_fil1, data_fil2, data_fil3))
    lb = np.concatenate([np.concatenate([file1, file2]), file3])
    
    
    X_train, X_test, y_train, y_test = train_test_split(
    data, lb , test_size=0.2, random_state=0)
    
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train , test_size=0.2, random_state=0)
    

#Load Dataset
df_train = pd.DataFrame()
num = 0
for lb in np.unique(y_train):
    if lb >=0 and lb<10:
        _, df = [x for _, x in data_csv1.groupby(data_csv1["img_lb"]==lb)]
    if lb >= 10 and lb < 20:
        _, df = [x for _, x in data_csv2.groupby(data_csv2["img_lb"]==lb)]
    if lb >= 20 and lb < 30:
        _, df = [x for _, x in data_csv3.groupby(data_csv3["img_lb"]==lb)]

    df['img_lb'] = num
    num = num + 1
    df_train = df_train.append(df)
df_train = df_train.set_index(np.arange(0,df_train.shape[0]))

df_val = pd.DataFrame()
num = 0
for lb in np.unique(y_val):
    if lb >=0 and lb<10:
        _, df = [x for _, x in data_csv1.groupby(data_csv1["img_lb"]==lb)]
    if lb >= 10 and lb < 20:
        _, df = [x for _, x in data_csv2.groupby(data_csv2["img_lb"]==lb)]
    if lb >= 20 and lb < 30:
        _, df = [x for _, x in data_csv3.groupby(data_csv3["img_lb"]==lb)]
    
    df['img_lb'] = num
    num = num + 1
    df_val = df_val.append(df)
df_val = df_val.set_index(np.arange(0, df_val.shape[0]))

df_test = pd.DataFrame()
num = 0
for lb in np.unique(y_test):
    if lb >=0 and lb<10:
        _, df = [x for _, x in data_csv1.groupby(data_csv1["img_lb"]==lb)]
    if lb >= 10 and lb < 20:
        _, df = [x for _, x in data_csv2.groupby(data_csv2["img_lb"]==lb)]
    if lb >= 20 and lb < 30:
        _, df = [x for _, x in data_csv3.groupby(data_csv3["img_lb"]==lb)]

    df['img_lb'] = num
    num = num + 1
    df_test = df_test.append(df)
df_test = df_test.set_index(np.arange(0, df_test.shape[0]))

data_train = Dataset(X_train, df_train , crop_size =CROP_SIZE)
data_val = Dataset(X_val, df_val , crop_size =CROP_SIZE)
data_test = Dataset(X_test, df_test , crop_size =CROP_SIZE)

train_dataloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)

model = UNet(3, 2).to(device)
print(model)

# Initialize the loss function
loss_fn = loss

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

transform_to_PIL = transforms.Compose([
      transforms.ToPILImage()
    ])

def train_loop(dataloader, model, loss_fn, optimizer, train_loss_arr, epoch, df, mask_id):
    size = len(dataloader.dataset)
    for batch, (X, y, cx, cy) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.byte()/255
        pred = model(X)
        idx = torch.randperm(X.shape[0])
        pred2 = pred[idx]
        loss =+ loss_fn(pred, pred2, lam = 1.0, beta=1)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        
        for indice in range(cx.shape[0]):

            mask = np.tile(np.array([255]), (720, 960)).reshape((720, 960))
            
            top_y = int(cy[indice]+100)
            top_x = int(cx[indice]+100)
            bot_x = int(cx[indice]-100)
            bot_y = int(cy[indice]-100)
            if (bot_x < 0):
                    bot_x = 0
            if (bot_y <0):
                    bot_y = 0
            if (top_y>=720):
                    top_y = 719
            if (top_x >= 960):
                    top_x = 959
            
            mask_id = mask_id +1
            _, im_th = cv2.threshold(np.array(transform_to_PIL(pred[indice][1])), 100, 255, cv2.THRESH_BINARY)
            bin_im = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE,    kernel = np.ones((3,3), np.uint8), iterations = 1)
            mask[int(bot_y):int(top_y), int(bot_x):int(top_x)]= bin_im[0:int(top_y-bot_y), 0:int(top_x - bot_x)]
            cv2.imwrite(PATH + 'mask' + '/' +"mask_id" + str(mask_id) + "_treino_epoch_" + str(epoch) + "_im_" + str(y[indice].item()) + "_cx_" + str(cx[indice].item()) + "_cy_" + str(cy[indice].item()) + ".png", mask)
            transform_to_PIL(pred[indice][1]).save(PATH + 'pred' + "/" +"pred_id" + str(mask_id) + "_treino_epoch_" + str(epoch) + "_im_" + str(y[indice].item()) + "_cx_" + str(cx[indice].item()) + "_cy_" + str(cy[indice].item()) + ".png")            
            df = df.append({"Mask_id": mask_id, "Im_id": y[indice], "Patch_Size": CROP_SIZE, "Centroide": tuple((cx[indice].item(), cy[indice].item())), "Ponto Inicial": tuple((cx[indice].item()-(CROP_SIZE/2), cy[indice].item() - (CROP_SIZE/2))) }, ignore_index=True)
    print("Train Loss: ", loss/int(size))
    df.to_excel(PATH + 'excel/' + "treino_epoch_" + str(epoch) + ".xlsx")
    train_loss_arr = np.concatenate((train_loss_arr, [loss/int(size)]))
    return train_loss_arr, mask_id

def test_loop(dataloader, model, loss_fn, test_loss_arr, test, epoch, df, mask_id):
 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    time_ = 0
    with torch.no_grad():
        for X, y, cx, cy in dataloader:
            X = X.byte()/255
            tin = time.time()
            pred = model(X)
            tfinal = time.time()
            dif_time = tfinal - tin
            idx = torch.randperm(X.shape[0])
            pred2 = pred[idx]
            test_loss += loss_fn(pred, pred2, lam = 1, beta=1).item()

            if test==False:
                str_path = "val_epoch_"
                str_path2 = "Val"
                if min(test_loss_arr) > test_loss:
                    torch.save(model.state_dict(), PATH + str(epoch) + 'best_model')
            else:
                str_path = "test"
                str_path2 = "Test"
            for indice in range(cx.shape[0]):

                mask = np.tile(np.array([255]), (720, 960)).reshape((720, 960))
            
                top_y = int(cy[indice]+100)
                top_x = int(cx[indice]+100)
                bot_x = int(cx[indice]-100)
                bot_y = int(cy[indice]-100)
                if (bot_x < 0):
                    bot_x = 0
                if (bot_y <0):
                    bot_y = 0
                if (top_y>=720):
                    top_y = 719
                if (top_x >= 960):
                    top_x = 959
                mask_id = mask_id +1
                _, im_th = cv2.threshold(np.array(transform_to_PIL(X[indice][1])), 100, 255, cv2.THRESH_BINARY)
                bin_im = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE,    kernel = np.ones((3,3), np.uint8), iterations = 1)
                mask[int(bot_y):int(top_y), int(bot_x):int(top_x)]= bin_im[0:int(top_y-bot_y), 0:int(top_x - bot_x)]
                cv2.imwrite(PATH + 'mask' + '/' + "mask_id" + str(mask_id) + str_path + "_epoch_" + str(epoch) + "_im_" + str(y[indice]) + "_cx_" + str(cx[indice]) + "_cy_" + str(cy[indice]) + ".png", mask)
                transform_to_PIL(X[indice][1]).save(PATH + 'pred' + '/' + "pred_id" + str(mask_id) + "_treino_epoch_" + str(epoch) + "_im_" + str(y[indice]) + "_cx_" + str(cx[indice]) + "_cy_" + str(cy[indice]) + ".png")
                df = df.append({"Mask_id": mask_id, "Im_id": y[indice].item(), "Patch_Size": CROP_SIZE, "Centroide": tuple((cx[indice].item(), cy[indice].item())), "Ponto Inicial": tuple((cx[indice].item()-(CROP_SIZE/2), cy[indice].item() - (CROP_SIZE/2))) }, ignore_index=True)
        time_ = time_ + dif_time
    df.to_excel(PATH + 'excel/' + str_path + str(epoch) + ".xlsx")
    print(str_path2 + " Loss: ", test_loss/int(size))
    print("Time: ", time_/int(size))
    test_loss_arr =  np.concatenate((test_loss_arr, [test_loss/int(size)]))
    return test_loss_arr, mask_id
            
train_loss_arr= np.array([0])
val_loss_arr = np.array([0])
test_losss_arr = np.array([0])
mask_id = 0
df = pd.DataFrame(columns= ["Mask_id", "Im_id", "Patch_Size" , "Centroide", "Ponto Inicial"])

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    t_loss, mask_id = train_loop(train_dataloader, model, loss_fn, optimizer, train_loss_arr, t, df, mask_id)
    val_loss, mask_id = test_loop(val_dataloader, model, loss_fn, val_loss_arr, False, t, df, mask_id)
    train_loss_arr = np.concatenate((train_loss_arr,t_loss))
    val_loss_arr = np.concatenate((val_loss_arr, val_loss))
test_loss = test_loop(test_dataloader, model, loss_fn, test_losss_arr, True, t, df, mask_id)
print("Done!")

plt.plot(train_loss_arr,'-o')
plt.plot(val_loss_arr,'-o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Val'])
plt.title('Train and Validation Loss')
 
plt.savefig(PATH + 'Train and Validation Loss.png')