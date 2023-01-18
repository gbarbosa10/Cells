# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:54:40 2023

@author: guiti
"""

import torch 

def loss(y, y2, mask = None, lam = 1.0, beta=1):
  '''
  Loss function for unsurpervised learning.
  y : torch tensor
  y2 : torch tensor
  mask : image from traditional methods that separates nucleus from background
  lam : lamda component that controls the product of one patch preiction times another
  beta = beta component that controls the size of the output, not to overpass the mask  
  '''
  l=0
  beta_yi_sum = 0
  y_ = torch.sigmoid(y)
  y_2 = torch.sigmoid(y2)
  log_yi = torch.log(torch.clamp(1.0 - y_2, 1e-07, 1.0))

  if mask is not None:
    beta_yi_sum += torch.sum(y_*(1 - mask))* beta

  l -= torch.sum(y_)
  l -= torch.sum(y_ * log_yi) * lam
  l -= beta_yi_sum

  return l / y_.shape[0]