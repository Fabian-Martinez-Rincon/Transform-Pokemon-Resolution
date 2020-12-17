import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Ruta raiz
PATH = '/content/drive/MyDrive/Transform-Pokemon-Resolution'
#Ruta de entrada
INPATH= PATH + "/Input Pokemon"
#Ruta de salida
OUTPATH = PATH + '/output Pokemon'
#Check Poins
CKPATH = PATH + '/checkpoints'

imgurls = !ls -1 "{INPATH}"

n = 151  #La cantidad de imagenes que voy a utilizar 
train_n = round(n * 0.80)#El porcentaje de images diferentes que voy a tener

#Lista randomizada
randurls = np.copy(imgurls)

np.random.seed(23)
np.random.shuffle(randurls)

#Particion train / test

tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]
#151 en la carpeta, 121 para entrenar y 30 diferentas
print(len(imgurls), len(tr_urls), len(ts_urls))

