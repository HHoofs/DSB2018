from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Multiply
from keras.models import Model
from keras.optimizers import Adam
try:
    from keras_implementation import generator
except:
    import generator

from keras.losses import binary_crossentropy

import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


from skimage.segmentation import mark_boundaries

from PIL import Image

import numpy as np
import os
import shutil

from skimage.segmentation import find_boundaries
from skimage.morphology import dilation
from scipy.ndimage.measurements import center_of_mass


def find_all_samples(path):
    all_samples = os.listdir(path)
    return all_samples


def create_mask(path, width, height):
    samples = find_all_samples(path)
    for sample in samples:
        sample_path = os.path.join(path, sample)
        sample_path_masks = os.path.join(sample_path, 'masks')
        masks = os.listdir(sample_path_masks)
        complete_mask = np.zeros((width, height), dtype=int)
        for mask in masks:
            with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                _mask = _mask.resize((width, height))
                _mask = np.array(_mask)
                complete_mask = np.maximum(complete_mask, _mask)
        os.mkdir(os.path.join(sample_path, 'mask'))
        mask_image = Image.fromarray(complete_mask.astype('uint8'), 'L')
        mask_image.save(os.path.join(sample_path, 'mask', '{}.png'.format(sample)))


def create_border_mask(path, width, height):
    samples = find_all_samples(path)
    for sample in samples:
        sample_path = os.path.join(path, sample)
        sample_path_masks = os.path.join(sample_path, 'masks')
        masks = os.listdir(sample_path_masks)
        complete_mask = np.zeros((width, height), dtype=int)
        for i, mask in enumerate(masks):
            with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                _array = np.array(_mask.resize((width, height)))
                _array = np.array(_array > 0, dtype=int) * (i + 1)
                complete_mask = np.add(complete_mask, _array)
        full_bounds = np.array(complete_mask, dtype=int)
        bound_array = find_boundaries(label_img = full_bounds, connectivity = 1, mode='outer', background=0)
        bound_array = np.array(bound_array, dtype=int)
        bound_array = bound_array * 255
        print(np.max(bound_array))

        # save image
        try:
            shutil.rmtree(os.path.join(sample_path, 'border_small'))
        except:
            pass
        os.mkdir(os.path.join(sample_path, 'border_small'))
        mask_image = Image.fromarray(bound_array.astype('uint8'), 'L')
        mask_image.save(os.path.join(sample_path, 'border_small', '{}.png'.format(sample)))


def foo(l, dtype=int):
    return map(dtype, l)


def create_border_hyper_mask(path, width, height):
    samples = find_all_samples(path)[1:]
    for sample in samples[:4]:
        sample_path = os.path.join(path, sample)
        sample_path_masks = os.path.join(sample_path, 'masks')
        masks = os.listdir(sample_path_masks)
        complete_mask = np.zeros((width, height), dtype=int)
        centers = []
        for i, mask in enumerate(masks):
            with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
                _array = np.array(_mask.resize((width, height)))
                centers.append(center_of_mass(_array))
                _array = np.array(_array > 0, dtype=int) * (i + 1)
                complete_mask = np.add(complete_mask, _array)
        full_bounds = np.array(complete_mask, dtype=int)
        bound_array = find_boundaries(label_img = full_bounds, connectivity = 1, mode='outer', background=0)
        bound_array = np.array(bound_array, dtype=int)
        bound_array = bound_array * - 1000
        full_bounds = full_bounds > 0
        full_bounds = np.array(full_bounds, dtype=int)
        print(centers)
        for coord in centers:
            print(coord)
            x, y = int(round(coord[0])), int(round(coord[1]))
            xx = [x-1,x,x+1]
            yy = [y-1,y,y+1]
            print(xx)
            print(yy)
            full_bounds[xx,yy] = 2
        full_bounds = np.add(full_bounds, bound_array)
        full_bounds[full_bounds<0] = -1
        full_bounds = full_bounds + 1
        full_bounds = full_bounds / 3
        complete_ = np.array(full_bounds, dtype=int)

        bound_array = complete_ * 255
        print(np.max(bound_array))

        # save image
        # shutil.rmtree(os.path.join(sample_path, 'trish'))
        os.mkdir(os.path.join(sample_path, 'trish'))
        mask_image = Image.fromarray(bound_array.astype('uint8'), 'L')
        mask_image.save(os.path.join(sample_path, 'trish', '{}.png'.format(sample)))


def mean_iou(y_true, y_pred):
    y_true = K.round(y_true)
    print(y_true)
    y_pred = K.round(y_pred)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
    return score


def mean_iou_border(y_true, y_pred):
    y_true = K.round(y_true)
    print(y_true)
    y_pred = K.round(y_pred)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
    return score


# def dice_coef(y_true, y_pred, smooth, thresh):
#     y_pred = y_pred > thresh
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
# def dice_loss(smooth, thresh):
#     def dice(y_true, y_pred):
#         return -dice_coef(y_true, y_pred, smooth, thresh)
#     return dice
#
#
# def custom_loss(y_true, y_pred, weight):
#     y_t = y_true * weight
#     y_p = y_pred * weight
#     de
#     return binary_crossentropy(y_t, y_p)


def custom_lo(weight):
    def cust(y_true, y_pred):
        return binary_crossentropy(y_true * weight, y_pred * weight)
    return cust


def create_model(filter_size = 10, drop_rate=.3):
    img_input = Input(shape=(256,256,1), name='input_image')

    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(img_input)
    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(2, 2)(conv1)

    conv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(2, 2)(conv2)

    conv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(2, 2)(conv3)

    conv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(conv4)
    drop4 = Dropout(drop_rate)(conv4)
    pool4 = MaxPooling2D(2, 2)(drop4)

    conv5 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same')(conv5)
    drop5 = Dropout(drop_rate)(conv5)

    # Upconvolutional layers
    uconv4 = Conv2DTranspose(filters=filter_size * 8, kernel_size=2, strides=2, activation='relu', padding='same')(drop5)
    uconc4 = concatenate([drop4, uconv4], axis=3)
    uconv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(uconc4)
    uconv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same')(uconv4)

    uconv3 = Conv2DTranspose(filters=filter_size * 4, kernel_size=2, strides=2, activation='relu', padding='same')(uconv4)
    uconc3 = concatenate([conv3, uconv3], axis=3)
    uconv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(uconc3)
    uconv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same')(uconv3)

    uconv2 = Conv2DTranspose(filters=filter_size * 2, kernel_size=2, strides=2, activation='relu', padding='same')(uconv3)
    uconc2 = concatenate([conv2, uconv2], axis=3)
    uconv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(uconc2)
    uconv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv2)

    uconv1 = Conv2DTranspose(filters=filter_size, kernel_size=2, strides=2, activation='relu', padding='same')(uconv2)
    uconc1 = concatenate([conv1, uconv1], axis=3)
    uconv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(uconc1)
    uconv1a = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1)
    uconv1a = Conv2D(filters=2, kernel_size=3, strides=1, activation='relu', padding='same')(uconv1a)

    pred_mask = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid', name='mask_out')(uconv1a)


    weight_input = Input(shape=(256,256,1), name='input_weight')

    loss_custom = custom_lo(weight_input)

    model = Model(inputs=[img_input,weight_input], outputs=[pred_mask])
    model.compile(loss=loss_custom, optimizer='rmsprop', metrics=['acc', mean_iou])
    return model



if __name__ == '__main__':
    # create_border_hyper_mask('/Users/HuCa/Documents/DSB2018/tessst', 256, 256)
    path_img = 'C:/Users/huubh/Documents/DSB2018_bak/img'
    # path_img = 'img'
    model_x2 = create_model()
    model_x2.summary()
    labels = os.listdir(path_img)[1:]
    training = labels[:32]
    # validation = labels[608:]
    print(len(training))
    # print(len(validation))
    training_generator = generator.DataGenerator(training, path_img,
                                                 rotation=True, flipping=True, zoom=1.25, batch_size = 16, dim=(256,256))

    prediction_generator_boundaries = generator.PredictDataGenerator2\
        (training[:2], path_img)
    # validation_generator = generator.DataGenerator(validation, path_img,
    #                                              rotation=True, flipping=True, zoom=False, batch_size = 31, dim=(256,256))
    for i in range(100):
        model_x2.fit_generator(generator=training_generator, epochs=2)
        predict = model_x2.predict_generator(prediction_generator_boundaries)
        print(np.max(predict))
        nico_ = generator.post_process_concat(training[:2], predict, threshold=4, bool=True)
        out_true = generator.post_process_original_size(nico_, path_img)
        for ids, out_arra in out_true.items():
            generator.plot_image_mask_hyper_out(ids, out_arra, path_img, i)


        # Save model
    # model_x2.save('model_dogeC.h5')