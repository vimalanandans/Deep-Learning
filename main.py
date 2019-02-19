import argparse
import keras
import ssl
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
import os

import numpy as np
#from IPython.display import Image
from keras.optimizers import Adam

from google_images_download import google_images_download


mobile = None
model_path = 'resources/mobinet_custom.h5'

def disable_ssl_certificate_check():
    """
    Needed for donwloading pre-trained models from internet
    :return:
    """
    ssl._create_default_https_context = ssl._create_unverified_context


def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def load_mobilenet():    
    global mobile
    exists = os.path.isfile(model_path)
    if exists :        
        mobile = load_model(model_path)
    else :   
        print ("loading pre-trained from web")
         # Disable checking for https certificate. Otherwise pre-trained model download will fail.
        disable_ssl_certificate_check()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        base_model = keras.applications.mobilenet.MobileNet()
        base_model = MobileNet(weights='imagenet',include_top=False) 
        mobile = custom_layers(base_model) 

    return mobile

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
    
def download_images():
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":"blue tit","limit":100,"print_urls":False,"format":"jpg", "size":">400*300"}
    paths = response.download(arguments)
    print(paths)
    arguments = {"keywords":"crow","limit":100,"print_urls":False, "format":"jpg", "size":">400*300"}
    paths = response.download(arguments)
    print(paths)
    return paths

def custom_layers(base_model):
    

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    for layer in model.layers:
        layer.trainable=False
    # or if we want to set the first 20 layers of the network to be non-trainable
    for layer in model.layers[:20]:
        layer.trainable=False
    for layer in model.layers[20:]:
        layer.trainable=True

    model.save(model_path)

    mobile.summary()

    return model

def train():

    model = load_mobilenet()
    
    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

    train_generator=train_datagen.flow_from_directory('./downloads',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

    step_size_train=train_generator.n//train_generator.batch_size
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10)

    model.save(model_path)

def infer_img(model, img_path):
    
    new_image = load_image(img_path)
    pred = model.predict(new_image)
    print ("#### Inference result ")
    print (pred)
    print("img_path {} >> blue_tit : {:.4f} crow : {:.4f}".format( img_path,pred[0][0],pred[0][1]));
    return

def infer():
    
    model = load_mobilenet()
    infer_img(model, 'crow.jpg')

    infer_img(model, 'blue_tit.jpg')
    
    return

def test_pretrained():
    disable_ssl_certificate_check()
    mobile = keras.applications.mobilenet.MobileNet()
    #Image(filename='German_Shepherd.jpg')
    preprocessed_image = prepare_image('German_Shepherd.jpg')
    predictions = mobile.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    print(results)

    print("")
    preprocessed_image = prepare_image('labrador1.jpg')
    predictions = mobile.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    print(results)

    print("")
    preprocessed_image = prepare_image('bulldog.jpg')
    predictions = mobile.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    # print(predictions)

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--phase', type=str, required=True, choices=['train', 'test', 'train_test','pretrained','img_dl'],
                        help='To train, test or both, pretrained -> run pretrained, img_dl -> download image for training ')                        
    return parser.parse_args()

def main():
    args = parse_args()  # Parse arguments

    cfg = dict()
    
    cfg['phase'] = str(args.phase)

    print(cfg['phase'])

    if cfg['phase'] in ('img_dl') : 
        download_images()

    if cfg['phase'] in ('pretrained'):
        test_pretrained()    
    
    if cfg['phase'] in ('train', 'train_test'):
        train()

    if cfg['phase'] in ('test', 'train_test'):
        infer()

    
    

if __name__ == '__main__':
    main()


