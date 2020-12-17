import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Ruta raiz
PATH = '/content/drive/MyDrive/Transform-Pokemon-Resolution'
#Ruta de entrada
INPATH= PATH + "/Input Pokemon"
#Ruta de salida
OUPATH = PATH + "/outputPoke"
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
#__________________________________________________________________
IMG_WIDTH = 256
IMG_HEIGHT = 256

#Reescalamos las imagenes
def resize(inimg, tgimg, height, width):
  inimg = tf.image.resize(inimg, [height, width])
  tgimg = tf.image.resize(tgimg, [height, width])
  return inimg, tgimg

#Normaliza el rango [-1, +1] la imagen
def normalize(inimg, tgimg):
  inimg = (inimg/127.5) - 1
  tgimg = (tgimg/127.5) - 1
  return inimg, tgimg

#Aumentacion de datos: Random Crop + Flip
def random_jitter(inimg, tgimg):
  inimg, tgimg = resize(inimg, tgimg, 286, 286)

  stacked_image = tf.stack([inimg, tgimg], axis = 0)
  cropped_image = tf.image.random_crop(stacked_image, size = [2, IMG_HEIGHT, IMG_WIDTH, 3])

  inimg, tgimg = cropped_image[0], cropped_image[1]
  
  if tf.random.uniform(()) > 0.5:

    inimg = tf.image.flip_left_right(inimg)
    tgimg = tf.image.flip_left_right(tgimg)

  return inimg, tgimg

def load_image(filename, augment=True):

  inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + '/' + filename)), tf.float32)[..., :3]
  tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUPATH + '/' + filename)), tf.float32)[..., :3]

  inimg, tgimg = resize(inimg, tgimg, IMG_HEIGHT, IMG_WIDTH)

  if augment:
    inimg, tgimg = random_jitter(inimg, tgimg)

  inimg, tgimg = normalize(inimg, tgimg)
  return inimg, tgimg

def load_train_image(filename):
  return load_image(filename, True)

def load_test_image(filename):
  return load_image(filename, False)

plt.imshow((load_train_image(randurls[0])[0]))