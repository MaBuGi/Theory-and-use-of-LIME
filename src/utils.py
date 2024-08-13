import numpy as np
from skimage.color import rgb2gray

from scipy.special import binom

#THE Following compute_functions have been forked from Garreau 2021 paper
def compute_alpha(d, nu=0.25, p=1):
    """
    Computes the alpha coefficients according to Proposition 1 of the paper.

    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter
        - p: number of distinct superpixels

    OUTPUT:
        alpha_p
    """
    s_values = np.arange(0,d+1)
    psi_values = compute_psi(s_values/d,nu)
    bin_values = binom(d-p,s_values)
    return np.dot(bin_values,psi_values) / 2**d

def compute_sigma_0(d,nu=0.25):
    """
    Computes \sigma_0 as in Proposition 2 of the supplementary.

    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter

    OUTPUT:
        \sigma_0

    """
    return (d-1)*compute_alpha(d,nu,2) + compute_alpha(d,nu,1)

def compute_sigma_1(d,nu=0.25):
    """
    Compute \sigma_1 as in Definition 2 of the paper.

    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter

    OUTPUT:
        \sigma_1
    """
    return -compute_alpha(d,nu,1)

def compute_sigma_2(d,nu=0.25):
    """
    Compute \sigma_2 as in Definition 2 of the paper.

    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter

    OUTPUT:
        \sigma_2
    """
    alpha_0 = compute_alpha(d,nu,0)
    alpha_1 = compute_alpha(d,nu,1)
    alpha_2 = compute_alpha(d,nu,2)
    return ((d-2)*alpha_0*alpha_2 - (d-1)*alpha_1**2 + alpha_0*alpha_1) / (alpha_1 - alpha_2)

def compute_sigma_3(d,nu=0.25):
    """
    Compute \sigma_3 as in Definition 2 of the paper.

    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter

    OUTPUT:
        \sigma_3
    """
    alpha_0 = compute_alpha(d,nu,0)
    alpha_1 = compute_alpha(d,nu,1)
    alpha_2 = compute_alpha(d,nu,2)
    return (alpha_1**2 - alpha_0*alpha_2) / (alpha_1 - alpha_2)

def compute_dencst(d,nu=0.25):
    """
    Compute c_d as in Definition 2 of the paper.

    INPUT:
        - d: number of superpixels
        - nu: bandwidth parameter

    OUTPUT:
        \sigma_1
    """
    alpha_0 = compute_alpha(d,nu,0)
    alpha_1 = compute_alpha(d,nu,1)
    alpha_2 = compute_alpha(d,nu,2)
    return (d-1)*alpha_0*alpha_2 - d*alpha_1**2 + alpha_0*alpha_1

#Compute psi values for weights
def compute_psi(t,nu=0.25):
  return np.exp(-np.square(1.0-np.sqrt(1.0 - t))/(2*nu**2))

#Convert tensor to image
def toImage(xj):
  return gray2rgb(torch.mul(xj,255.0)).reshape(28,28,3)

#Trudge image
def trudge_image(xj_image, segments, s_i, color=(255, 0, 0)):
  xj_copy = xj_image.copy()
  indexes = np.where(s_i == 0)
  for i in indexes[0]:
    xj_copy[segments == i] = color
  return xj_copy

#Generate fixed design Z, (small d)
def all_comb(d):
  lenst ='{0:0'+str(d)+'b}'
  return np.flip([[int(i) for i in list(lenst.format(j))] for j in range(2**d)], axis=0)

#Trudge tensor
def trudge_input(x, segments, turn_list):
  x1 = x.clone().detach()
  indexes = np.where(turn_list == 0)
  for i in indexes[0]:
    x1[segments == i] = 0.0
  return x1

#Probability distribution to argmax label
def toValue(out_xt):
  return np.argmax(a=out_xt[0].detach().numpy())

# Given design Z trudge sentence
def trudge(sentence, Z):
  words = sentence.split(" ")
  texts = []
  for i in range(Z.shape[0]):
    trudged = ""
    for j in range(Z.shape[1]-1):
      if bool(Z[i,j]):
        trudged += words[j] + " "

    if bool(Z[i,Z.shape[1]-1]):
        trudged += words[Z.shape[1]-1]

    if(len(trudged)>0):
      if(trudged[-1] == " "):
        trudged = trudged[:-1]
    texts.append(trudged)
  return texts

# Appends interaction columns to design Z
def append_interaction_columns(arr1):
  arr=arr1
  for i in range(arr1.shape[1]):
    for j in range(i+1, arr1.shape[1]):
      arr = np.concatenate(
        (arr, np.expand_dims(
                np.multiply(arr[:, i], arr[:, j]), axis=1)), axis=1
        )
  return arr
#Returns string list of words and mutual interactions
def include_interactions(words):
  ls = list(words)
  for i in range(len(words)):
    for j in range(i+1, len(words)):
      add = words[i]+"-"+words[j]
      ls.append(add)
  return ls
