from keras.models import *
from keras.layers import *

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder
# from .efficientnet import EfficientNetB3
import efficientnet.keras as efn 

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def unet_mini(n_classes, input_height=360, input_width=480):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)

    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv2)

    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv3), conv2], axis=MERGE_AXIS)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv4), conv1], axis=MERGE_AXIS)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv5)

    o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
               padding='same')(conv5)

    model = get_segmentation_model(img_input, o)
    model.model_name = "unet_mini"
    return model


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)

    return model





def _efficient_unt(n_classes, encoder, l1_skip_conn=True, input_height=480,input_width=640):
    base_model = encoder(input_shape=(input_height,input_width,3),include_top=False,weights='noisy-student')
    for layer in base_model.layers:
        layer._name =layer.name +str("_2")
        print(layer._name)
    base_model.summary()
    # Layer freezing?
    for layer in base_model.layers: layer.trainable = True
    o = base_model.output
    img_input = base_model.input
    f1 = base_model.get_layer('stem_conv').output 
    f2 = base_model.get_layer('block2a_dwconv').output
    f3 = base_model.get_layer('block3a_dwconv').output
    f4 = base_model.get_layer('block4a_dwconv').output
    # f5 = base_model.get_layer('block5a_dwconv').output
    

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f4], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)


    # def upproject(tensor,filters,name,concat_with):
    #     up_i = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(tensor)

    #     # up_i = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(up_i)
    #     up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
            

    #     up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
    #     # up_i = LeakyReLU(alpha=0.2)(up_i)
          
    #     return up_i

    # base_model = encoder(input_shape=(input_height,input_width,3),include_top=False,weights='noisy-student')
    # base_model_output_shape = base_model.layers[-1].output.shape
    # decode_filters = int(int(base_model_output_shape[-1])/2)
    # decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)
    # decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='block4a_dwconv')
    # decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='block3a_dwconv')
    # decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='block2a_dwconv')
    # decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='stem_conv')
    # conv3 = Conv2D(filters=n_classes, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

    # model = get_segmentation_model(base_model.input, conv3)
    # model.summary()
    return model
    

def unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "unet"
    return model


def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "vgg_unet"
    return model


def resnet50_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "resnet50_unet"
    return model
def EfficientNetB3_unet(n_classes,input_height,input_width):
    
    model = _efficient_unt(n_classes,efn.EfficientNetB3,input_height=input_height,input_width=input_width)
    
    return model

def mobilenet_unet(n_classes, input_height=224, input_width=224,
                   encoder_level=3):

    model = _unet(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width)
    model.model_name = "mobilenet_unet"
    return model


if __name__ == '__main__':
    m = unet_mini(101)
    m = _unet(101, vanilla_encoder)
    # m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
    m = _unet(101, get_vgg_encoder)
    m = _unet(101, get_resnet50_encoder)
