#!/usr/bin/env python
# coding: utf-8

# wczytywanie zdjęć

# In[1]:


from PIL import Image
import glob
import numpy as np
from matplotlib import pyplot
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as dst
import matplotlib.pyplot as matlab
import matplotlib.mlab as mlab
image_list = []
for filename in glob.glob('C:/Users/akomo/Desktop/testmgr/*.jpeg'): #ścieżka do folderu z sekwencją
    print(filename)
    img = np.asarray(Image.open(filename))
    image_list.append(img)


# zapisywanie wszystkiego do np.array

# In[2]:


new_images = [img.copy() for img in image_list]
abc = np.stack(new_images, axis=0)


# In[3]:


wi_pocz = 197
wi_kon = 351
wj_pocz = 68
wj_kon = 680


# In[4]:


from scipy.stats import norm
import imageio
import cv2


# #zdefiniowanie funkcji

# In[5]:


def Mstep(dataSet,W):
    N= np.shape(dataSet)
    K = np.size(W,1)
    N_k = np.sum(W,0)
    Alpha = N_k/np.sum(N_k)
    data.T.dot(W).dot(np.diag(np.reciprocal(N_k)))
    Sigma = sd(W,data,Mu)
    return Alpha,Mu,Sigma


# In[6]:


def Estep(dataSet,Alpha,Mu,Sigma):

    N = np.size(dataSet,0)
    K = np.size(Alpha)
    W = np.zeros([N,K])
    W[:,0] = Alpha[0]*norm.pdf(dataSet.T,Mu[0],Sigma[0])
    W[:,1] = Alpha[1]*norm.pdf(dataSet.T,Mu[1],Sigma[1])
    W = W*np.reciprocal(np.sum(W,1)[None].T)
    return W


# In[7]:


def logLike(dataSet,Alpha,Mu,Sigma):
    K = len(Alpha)
    N = len(dataSet)
    P = np.zeros([N,K])
    P[:,0] = norm.pdf(dataSet.T,Mu[0],Sigma[0])
    P[:,1] = norm.pdf(dataSet.T,Mu[1],Sigma[1])
    return np.sum(np.log(P.dot(Alpha)))


# In[8]:


def sd(W,data,Mu):
    pom0 = [np.power((W[i,0] * data[i])-Mu[0],2) for i in range(len(data))]
    pom1 = [np.power((W[i,1] * data[i])-Mu[1],2) for i in range(len(data))]
    sd0 = np.sum(pom0)/len(data)
    sd1 = np.sum(pom1)/len(data)
    return np.asarray([sd0,sd1])


# poniżej obliczam odleglości mahalanobisa i uzupełniam np.array - u mnie trwa to do 10 minut

# In[10]:


dist_array = np.zeros((len(new_images),480,720)) #ones


# In[11]:


for wj in range(wj_pocz,wj_kon):
    for wi in range(wi_pocz,wi_kon):
        print("new")
        lst = abc[:,wi,wj,:]
        Mu = np.mean(lst,axis=0)       
        Sigma = np.cov(lst.T)
        print(datetime.now())
        if ((np.absolute(np.linalg.det(Sigma[:,:]))>0.5)):
            Sigma_inv = np.linalg.inv(Sigma)
            distance = [dst.mahalanobis(x,Mu,Sigma_inv) for x in lst]
            print(datetime.now())
            dist_array[:,wi,wj] = distance
        else:
            distance= [0 for x in lst]
            print("zero cov",datetime.now())
    print(wj)


# ## EM

# In[12]:


for wj in range(wj_pocz,wj_kon):
    print("start",wj," czas ",datetime.now())
    czas = datetime.now()
    for wi in range(wi_pocz,wi_kon):
        data = dist_array[:,wi,wj]
        weights_for = (data-min(data))/(max(data)-min(data))
        weights_for = [np.round(min(1.0,x)) for x in weights_for]
        if np.sum(weights_for)==0:
            new_rgb = [0 for w in W] #normalnie 1
            dist_array[:,wi,wj] = new_rgb
        else:
            weights_back = [1-x for x in weights_for] #wagi tła (pstwa)
            W = zip(weights_for,weights_back)
            W = [list(elem) for elem in W]
            N = len(data) #liczba zdjęc
            K=2 #tlo/sylwetka
        ####################DO POPRAWY
            x=np.zeros([N,K])
            for k in range(K):
                for i in range(N):
                    x[i,k] = W[i][k]
            W=x
        ###############################
            N=np.shape(distance)
            N_k = np.sum(W,0)
            if (np.std(data)<0.85): #proba bylo 0.7
                new_rgb = [0 for w in W] #bylo 1
                dist_array[:,wi,wj] = new_rgb
            else:
            #obliczam parametry obu rozkladow w mieszaninie
                Alpha = N_k/np.sum(N_k)
                Mu = data.T.dot(W).dot(np.diag(np.reciprocal(N_k)))
                Sigma = sd(W,data,Mu)
                iter = 0
                prevll = -999999
                while(True):
                    W = Estep(data,Alpha,Mu,Sigma)
                    Alpha,Mu,Sigma = Mstep(data,W)
                    ll_train = logLike(data,Alpha,Mu,Sigma)
                    iter = iter + 1
                    if(iter>100 or abs(ll_train - prevll)< 0.01): #150
                        break
                    prevll = ll_train
                    #print(iter)
                new_rgb = [1 if w[0]>0.5 else 0 for w in W] #normalnie 0,1 teraz
                dist_array[:,wi,wj] = new_rgb
                #print("new_rgb ",datetime.now())
                print(iter)
                print('0: ',new_rgb.count(0))
                #print('255: ',new_rgb.count(1))
                #print("FALSE",wi,wj)
    print(datetime.now()-czas)
    print("koniec",wj)
    print("\n")


# zapisuję zdjęcia do plików (trzeba stworzyć folder)

# In[14]:


for x in range(len(new_images)):
    im = dist_array[x,:,:]
    im[im == 1.0] = 255
    im = im.astype(np.uint8)
    nazwa = "C:/Users/akomo/Desktop/output/"+str(x)+".jpeg"
    imageio.imwrite(nazwa, im)
    print(x)


# Poniższe kroki służą kolejno do:  
# *  usuwania szumów
# *  wygładzenia krawędzi (choć może to zaburzyć ewentualną analizę chodu - zdjęcia są jednak mniej poszarpane, więc wykorzystałem te metodę

# W poniższych kodach trzeba na sztywno podmienić ściezki - foldery muszą być utworzone ręcznie - to robocza wersja

# In[53]:


def undesired_objects (image,x):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    nazwa = "C:/Users/akomo/Desktop/Cleaned/"+str(x)+".jpeg"
    cv2.imwrite(nazwa, img2)


# In[58]:


for x in range(len(new_images)):
    print(x)
    nazwa = 'C:/Users/akomo/Desktop/output/'+str(x)+'.jpeg'
    img = cv2.imread(nazwa,cv2.IMREAD_GRAYSCALE)
    img = undesired_objects(img,x)


# In[62]:


for x in range(len(new_images)):
    print(x)
    nazwa = 'C:/Users/akomo/Desktop/Cleaned/'+str(x)+'.jpeg'
    img = cv2.imread(nazwa)
    img2 =  cv2.medianBlur(img,5)
    nazwa = 'C:/Users/akomo/Desktop/Blurred/'+str(x)+'.jpeg'
    cv2.imwrite(nazwa, img2)

