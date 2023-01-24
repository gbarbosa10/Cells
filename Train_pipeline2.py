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
from PIL import Image

learning_rate = 5e-3
batch_size = 32
epochs = 30

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

SEPARATE_FILE_IN_TEST = True
PATH = '/home/vm/mariana/'
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
    df_train = pd.concat([df_train, df], ignore_index = True)
df_train = df_train.set_index(np.arange(0, df_train.shape[0]))

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
    df_val = pd.concat([df_val, df], ignore_index = True)
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
    df_test = pd.concat([df_test, df], ignore_index = True)
df_test = df_test.set_index(np.arange(0, df_test.shape[0]))


data_train = Dataset(X_train.to(device), df_train , crop_size =CROP_SIZE)
data_val = Dataset(X_val.to(device), df_val , crop_size =CROP_SIZE)
data_test = Dataset(X_test.to(device), df_test , crop_size =CROP_SIZE)

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

def find_cts_image(mask):
    contours_all, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    return contours_all

def draw_contours_image(contours, img_orig):

 # Draw Contours and Remove Objects based on Min and Max Nuclei Area
    img = img_orig.copy()
    for i in range(len(contours)):
        cv2.drawContours(img, contours, i, (30,255,255), 2, cv2.LINE_8)
    return img

def train_loop(dataloader,  dataset, model, loss_fn, optimizer, epoch):
    df = pd.DataFrame(columns= ["Mask_id", "Im_id", "Patch_Size" , "Centroide_x", "Centroide_y", "Ponto Inicial_x", "Ponto Inicial_y"])
    size = len(dataloader.dataset)
    mask_id = 0
    d = pd.DataFrame(columns= ["Mask_id", "Im_id", "Patch_Size" , "Centroide", "Ponto Inicial"])
    if not os.path.exists(PATH + 'mask/train/epoch_' + str(epoch)):
        os.mkdir(PATH + 'mask/train/epoch_' + str(epoch))
    if not os.path.exists(PATH + 'contour/train/epoch_' + str(epoch)):
        os.mkdir(PATH + 'contour/train/epoch_' + str(epoch))
    if not os.path.exists(PATH + 'pred/train/epoch_' + str(epoch)):
        os.mkdir(PATH + 'pred/train/epoch_' + str(epoch))

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
            mask = np.tile(np.array([0]), (720, 960)).reshape((720, 960))
            
            top_y = int(cy[indice]+CROP_SIZE/2)
            top_x = int(cx[indice]+CROP_SIZE/2)
            bot_x = int(cx[indice]-CROP_SIZE/2)
            bot_y = int(cy[indice]-CROP_SIZE/2)
            bot_patch_x = 0
            bot_patch_y = 0
            top_patch_y = CROP_SIZE
            top_patch_x = CROP_SIZE
            if (bot_x < 0):
                    bot_patch_x = abs(bot_x)
                    bot_x = 0
            if (bot_y <0):
                    bot_patch_y = abs(bot_y)
                    bot_y = 0
            if (top_y>=720):
                    top_patch_y = CROP_SIZE - (top_y - 719)
                    top_y = 719
            if (top_x >= 960):
                    top_patch_x = CROP_SIZE - (top_x - 959)
                    top_x = 959
            
            mask_id = mask_id +1
            _, im_th = cv2.threshold(np.array(transform_to_PIL(pred[indice][1])), 100, 255, cv2.THRESH_BINARY)
            bin_im = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel = np.ones((3,3), np.uint8), iterations = 1)
            mask[int(bot_y):int(top_y), int(bot_x):int(top_x)] = bin_im[bot_patch_y:top_patch_y, bot_patch_x:top_patch_x]
            counters = find_cts_image(np.array(mask.reshape((720, 960)), dtype=np.uint8))
            og_im = np.array(dataset._get_image(y[indice].item()))
            im_final = draw_contours_image(counters , og_im)
            transform_to_PIL(im_final).save(PATH + "contour/train/epoch_" + str(epoch) +"/contour_id_" + str(mask_id) + ".png")
            cv2.imwrite(PATH + "mask/train/epoch_" + str(epoch) + "/mask_id_" + str(mask_id) + ".png", mask)
            transform_to_PIL(pred[indice][1]).save(PATH + 'pred/train/epoch_' + str(epoch) + "/pred_id_" + str(mask_id) + ".png" )
            dict = {"Mask_id": 0, "Im_id": 0, "Patch_Size": 0, "Centroide_x": 0, "Centroide_y": 0, "Ponto Inicial_x": 0, "Ponto Inicial_y": 0}
            d = pd.DataFrame(data= dict, columns= ["Mask_id", "Im_id", "Patch_Size" , "Centroide_x", "Centroide_y", "Ponto Inicial_x", "Ponto Inicial_y"], index=[0])
            d["Mask_id"] = mask_id
            d["Im_id"] = y[indice].item()
            d["Patch_Size"] = CROP_SIZE
            d["Centroide_x"] = cx[indice].item()
            d["Centroide_y"] = cy[indice].item()
            d["Ponto Inicial_x"] = cx[indice].item()-(CROP_SIZE/2)
            d["Ponto Inicial_y"] = cy[indice].item() - (CROP_SIZE/2)

                  
            df = pd.concat([df, d], ignore_index = True)
    	    
    print("Train Loss: ", loss/int(size))
    df.to_excel(PATH + 'excel/' + "train_epoch_" + str(epoch) + ".xlsx")
    train_loss = loss/int(size)
    return train_loss

def test_loop(dataloader, dataset , model, loss_fn, test, epoch):
    df = pd.DataFrame(columns= ["Mask_id", "Im_id", "Patch_Size" , "Centroide_x", "Centroide_y", "Ponto Inicial_x", "Ponto Inicial_y"])
    
    if test==False:
       str_path = "val_epoch_"
       str_path2 = "val" 
    
    else:
       str_path2 = "test"
       str_path = "test"
     
    if not os.path.exists(PATH + 'mask/' + str_path2 + '/epoch_' + str(epoch)):
       os.mkdir(PATH + 'mask/' + str_path2 + '/epoch_' + str(epoch))
    if not os.path.exists(PATH + 'contour/' + str_path2 + '/epoch_' + str(epoch)):
       os.mkdir(PATH + 'contour/' + str_path2 + '/epoch_' + str(epoch))
    if not os.path.exists(PATH + 'pred/' + str_path2 + '/epoch_' + str(epoch)):
       os.mkdir(PATH + 'pred/' + str_path2 + '/epoch_' + str(epoch))

    mask_id = 0
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
            time_ = time_ + dif_time
            for indice in range(cx.shape[0]):

                mask = np.tile(np.array([0]), (720, 960)).reshape((720, 960))
            
                top_y = int(cy[indice]+CROP_SIZE/2)
                top_x = int(cx[indice]+CROP_SIZE/2)
                bot_x = int(cx[indice]-CROP_SIZE/2)
                bot_y = int(cy[indice]-CROP_SIZE/2)
                bot_patch_x = 0
                bot_patch_y = 0
                top_patch_y = CROP_SIZE
                top_patch_x = CROP_SIZE

                if (bot_x < 0):
                    bot_patch_x = abs(bot_x)
                    bot_x = 0
                if (bot_y <0):
                    bot_patch_y = abs(bot_y)
                    bot_y = 0
                if (top_y>=720):
                    top_patch_y = CROP_SIZE - (top_y - 719)
                    top_y = 719
                if (top_x >= 960):
                    top_patch_x = CROP_SIZE - (top_x - 959)
                    top_x = 959
                mask_id = mask_id +1
                _, im_th = cv2.threshold(np.array(transform_to_PIL(X[indice][1])), 100, 255, cv2.THRESH_BINARY)
                bin_im = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE,    kernel = np.ones((3,3), np.uint8), iterations = 1)
                mask[int(bot_y):int(top_y), int(bot_x):int(top_x)]= bin_im[bot_patch_y:top_patch_y, bot_patch_x:top_patch_x]
                counters = find_cts_image(np.array(mask.reshape((720, 960)), dtype=np.uint8))
                og_im = np.array(dataset._get_image(y[indice].item()))
                im_final = draw_contours_image(counters , og_im)
                transform_to_PIL(im_final).save(PATH + "contour/" + str_path2 + "/epoch_" + str(epoch) + "/contour_id_" + str(mask_id) + ".png")
                cv2.imwrite(PATH + "mask/" + str_path2 + "/epoch_" + str(epoch) + "/mask_id" + str(mask_id) + ".png", mask)
                transform_to_PIL(X[indice][1]).save(PATH + 'pred/' + str_path2 + "/epoch_" + str(epoch) + "/pred_id_" + str(mask_id) + ".png")
                dict = {"Mask_id": 0, "Im_id": 0, "Patch_Size": 0, "Centroide_x": 0, "Centroide_y": 0, "Ponto Inicial_x": 0, "Ponto Inicial_y": 0}
                d = pd.DataFrame(data = dict , columns= ["Mask_id", "Im_id", "Patch_Size" , "Centroide_x", "Centroide_y", "Ponto Inicial_x", "Ponto Inicial_y"], index=[0])
                d["Mask_id"] = mask_id
                d["Im_id"] = y[indice].item()
                d["Patch_Size"] = CROP_SIZE
                d["Centroide_x"] = cx[indice].item()
                d["Centroide_y"] = cy[indice].item()
                d["Ponto Inicial_x"] = int(cx[indice].item()-(CROP_SIZE/2))
                d["Ponto Inicial_y"] = int(cy[indice].item() - (CROP_SIZE/2))
                  
                df = pd.concat([df, d], ignore_index = True)
    
    torch.save(model.state_dict(), PATH + 'best_model_' + str(epoch))  	 
    df.to_excel(PATH + 'excel/' + str_path + str(epoch) + ".xlsx")
    print(str_path2 + " Loss: ", test_loss/int(size))
    print("Time: ", time_/int(size))
    test_loss =  test_loss/int(size)
    return test_loss, time_/int(size)
            
train_loss_arr = list([])
val_loss_arr = list([])
test_loss_arr = list([])
time_list = list([])

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    t_loss = train_loop(train_dataloader, data_train , model, loss_fn, optimizer, t)
    val_loss, time__ = test_loop(val_dataloader, data_val , model, loss_fn, False, t)
    time_list.append(time__)
    train_loss_arr.append(t_loss)
    val_loss_arr.append(val_loss)
test_loss, test_time = test_loop(test_dataloader, data_test , model, loss_fn, True, t)
print("Done!")

result_file = open(PATH + 'Test_results.txt', 'a')
result_file.writelines("Test Loss: " + str(test_loss))
result_file.writelines(" Test Time: " + str(test_time))
result_file.close()

epocas = np.arange(epochs)

fig1 = plt.figure()
fig1, ax = plt.subplots(figsize=(5, 3))
ax.plot(epocas, train_loss_arr,'-o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(['Train'])
ax.set_title('Train Loss')

fig1.savefig(PATH + 'Train Loss.png')

fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.plot(epocas, val_loss_arr,'-o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(['Val'])
ax2.set_title('Validation Loss')

fig2.savefig(PATH + 'Validation Loss.png')

fig3, ax3 = plt.subplots(figsize=(5, 3))
ax3.plot(epocas, time_list,'-o')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Time')
ax3.legend(['Val'])
ax3.set_title('Time for Patch')

fig3.savefig(PATH + 'Time for Patch.png')

