# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:56:09 2023

@author: guiti
"""

import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
import torch
import numpy as np

class Dataset:
  def __init__(self, data_img, dataframe,  mask_img = None, crop_size = 64):
    '''
    data_img: Cell image. A nd_array of 3 dimensions. For multi-channel input, the channel is the first axis.
    dataframe: A pandas type file with: 
              centroids_x: Array of coordinates for the nucleus in the horizontal axis of an image
              centroids_y: Array of coordinates for the nucleus in the vertical axis of an image
              img_lb: A diferent label is associated with each image. 
    mask_img: binary image with backgound separated from nucleus
    crop_size: Length of the side of each patch created
    '''

    transform_to_tensor = transforms.Compose([
      transforms.PILToTensor()
    ])

    transform_to_PIL = transforms.Compose([
      transforms.ToPILImage()
    ])

    self.lb_list = np.arange(0, np.unique(dataframe['img_lb'].to_numpy()).size)

    self.image_to_tensor = transform_to_tensor
    self.tensor_to_image = transform_to_PIL
    
    self.centroid_list = dataframe
    self._crop_size = crop_size
    self._imgs = data_img
    self._mask = mask_img

    self._create_im_patches()


  def __len__(self):
      return len(self._patches)

  def __getitem__(self, idx):
      patch = self._patches[idx]
      lb = int(self.centroid_list['img_lb'].iloc[idx])
      cx = int(self.centroid_list['centroid_x'].iloc[idx])
      cy = int(self.centroid_list['centroid_y'].iloc[idx])
      return patch, lb, cx , cy

  def get_patch_centroids(self,idx):
    centroid_list = self.centroid_list

    x, y = centroid_list['centroid_x'].iloc[idx], centroid_list['centroid_y'].iloc[idx]
    return x, y 

  def get_image_centroids(self, image_lb):
    centroid_list = self.centroid_list

    x, y = centroid_list['centroid_x'].where(centroid_list['img_lb']==image_lb).to_numpy(), centroid_list['centroid_y'].where(centroid_list['img_lb']==image_lb).to_numpy()
    x = x[np.logical_not(np.isnan(x))]
    y = y[np.logical_not(np.isnan(y))]
    return x, y 

  def _create_im_patches(self):
    # separates the input image into square patches according to the nucleus locations. This way the cells are segmented into specific patches

    data_img_ = self._imgs
    crop_size_ = self._crop_size
    sq_side = self._crop_size
    lb_list_ = self.lb_list

    centroid_x_, centroid_y_ = self.get_image_centroids(lb_list_[0])
    patches = functional.resized_crop(data_img_[0], int(centroid_y_[0]) - int((sq_side/2)), int(centroid_x_[0]) - int((sq_side/2)), crop_size_, crop_size_, crop_size_).reshape(1, data_img_[0].size()[0], crop_size_, crop_size_)
    for ind in lb_list_:
      centroid_x_, centroid_y_ = self.get_image_centroids(ind)
     
      for ind2 in range(centroid_x_.size): 

        crop_im = functional.resized_crop(data_img_[int(ind)], int(centroid_y_[ind2]) - int((crop_size_/2)), int(centroid_x_[ind2]) - int((crop_size_/2)), crop_size_, crop_size_, crop_size_)
        patches = torch.cat((patches, crop_im.reshape(1, crop_im.size()[0], crop_im.size()[1], crop_im.size()[2])))
        
    self._patches = patches[1:, :, :, :]   

  def _show_labels(self):
    print(self.lb_list)

  def _show_image(self, lb):
    image = self._imgs[lb]
    transform = self.tensor_to_image
    transform(image).show()

  def _get_image(self, lb):
    image = self._imgs[lb]
    transform = self.tensor_to_image
    return transform(image)

  def _show_patches(self, lb):
    centroid_list = self.centroid_list
    marker = centroid_list['img_lb'].where(centroid_list['img_lb']==lb).to_numpy()
    patches_ = self._patches[np.logical_not(np.isnan(marker)), :, :, :]
    transform = self.tensor_to_image
    for patch in patches_:
      transform(patch).show()

  def show_image_centroids(self, image_lb):
    centroid_list = self.centroid_list

    x, y = centroid_list['centroid_x'].where(centroid_list['img_lb']==image_lb).to_numpy(), centroid_list['centroid_y'].where(centroid_list['img_lb']==image_lb).to_numpy()
    x = x[np.logical_not(np.isnan(x))]
    y = y[np.logical_not(np.isnan(y))]
    print('Coordenate X: ', x)
    print("Coordenate Y:" , y) 

  @property
  def crop_size(self):
    return self._crop_size