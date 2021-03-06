import numpy as np
import keras
import os
from PIL import Image, ImageOps
from scipy.ndimage import affine_transform
from scipy.ndimage import morphology as morph
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imresize
from skimage import morphology
from skimage.color import label2rgb
import matplotlib
matplotlib.rcParams.update({'font.size': 18, 'figure.edgecolor': 'black'})


def sample_x_y(samples, path, x_shape=(256,256), y_shape=(256,256), mirror_edges=0):

    # TODO: Shuffle

    out_shape = (x_shape[0] + mirror_edges, x_shape[1] + mirror_edges)

    X = np.empty((len(samples), *out_shape, 1))
    Y = np.empty((len(samples), *y_shape, 1))

    for i, sample in enumerate(samples):
        with Image.open(os.path.join(path, sample, 'images', '{}.png'.format(sample))) as x_img:
            x_img = x_img.convert(mode='L')
            x_img = ImageOps.autocontrast(x_img)
            x_arr = np.array(x_img) / 255
            x_arr = np.expand_dims(x_arr, axis=2)
            X[i,] = x_arr
        with Image.open(os.path.join(path, sample, 'mask', '{}.png'.format(sample))) as y_img:
            y_arr = np.array(y_img) / 255
            y_arr = np.expand_dims(y_arr, axis=2)
            Y = y_arr

    return X, Y, samples


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, batch_size=4, dim=(256,256), n_channels=1, shuffle=True,
                 rotation=False, flipping=False, zoom=False, mirror_edges=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path = path
        self.rotation = rotation
        self.flipping = flipping
        self.zoom = zoom
        self.mirror_edges = mirror_edges
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.mirror_edges:
            out_shape = (self.dim[0] + self.mirror_edges, self.dim[0] + self.mirror_edges)
        else:
            out_shape = self.dim

        X_image = np.empty((self.batch_size, *out_shape, self.n_channels))
        X_weigh = np.empty((self.batch_size, *out_shape, self.n_channels))

        X_d = {'input_image': X_image, 'input_weight': X_weigh}

        Y = np.empty((self.batch_size, *self.dim, self.n_channels))


        if self.rotation:
            rot = np.random.choice([0, 90, 180, 270], self.batch_size)
        else:
            rot = np.zeros((self.batch_size))

        if self.flipping:
            flip = np.random.choice([True, False], (2,self.batch_size))
        else:
            flip = np.zeros((2, self.batch_size), dtype=bool)


        if self.zoom:
            zoom_l = np.random.choice([True, False, False], self.batch_size)
            zoom_o = [False] * self.batch_size
            for i, zo in enumerate(zoom_l):
                if zo:
                    zoom_factor = random.uniform(1, 1/self.zoom)
                    size = np.floor(self.dim[0]*zoom_factor)
                    x_co, y_co = np.random.randint(0, self.dim[0] - size, 2)
                    zoom_o[i] = (x_co, y_co, int(x_co + size), int(y_co + size))

        else:
            zoom_o = np.zeros((self.batch_size), dtype=bool)

        # Generate data
        for i, sample in enumerate(list_IDs_temp):

            with Image.open(os.path.join(self.path, sample, 'images', '{}.png'.format(sample))) as x_img:
                x_img = x_img.resize(self.dim)
                if zoom_o[i]:
                    x_img = x_img.crop((zoom_o[i][0], zoom_o[i][1], zoom_o[i][2], zoom_o[i][3]))
                    x_img = x_img.resize(self.dim)
                x_img = x_img.rotate(rot[i])
                if flip[0,i]:
                    x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
                if flip[1,i]:
                    x_img = x_img.transpose(Image.FLIP_TOP_BOTTOM)
                x_img = x_img.convert(mode='L')
                x_img = ImageOps.autocontrast(x_img)

                x_arr = np.array(x_img) / 255
                if self.mirror_edges:
                    x_arr = affine_transform(x_arr, [1,1], offset=[self.mirror_edges/2, self.mirror_edges/2],
                                             output_shape=out_shape, mode='mirror')
                x_arr = np.expand_dims(x_arr, axis=2)

                X_d['input_image'][i,] = x_arr


            with Image.open(os.path.join(self.path, sample, 'mask', '{}.png'.format(sample))) as y_img:
                y_img = y_img.resize(self.dim)
                if zoom_o[i]:
                    y_img = y_img.crop((zoom_o[i][0], zoom_o[i][1], zoom_o[i][2], zoom_o[i][3]))
                    y_img = y_img.resize(self.dim)
                y_img = y_img.rotate(rot[i])
                if flip[0,i]:
                    y_img = y_img.transpose(Image.FLIP_LEFT_RIGHT)
                if flip[1,i]:
                    y_img = y_img.transpose(Image.FLIP_TOP_BOTTOM)
                y_arr_s = np.array(y_img) / 255
                y_arr = np.expand_dims(y_arr_s, axis=2)
                y_arr_store = y_arr


            with Image.open(os.path.join(self.path, sample, 'smashing_border', '{}.png'.format(sample))) as weight_img:
                weight_img = weight_img.resize(self.dim)
                if zoom_o[i]:
                    weight_img = weight_img.crop((zoom_o[i][0], zoom_o[i][1], zoom_o[i][2], zoom_o[i][3]))
                    weight_img = weight_img.resize(self.dim)
                weight_img = weight_img.rotate(rot[i])
                if flip[0,i]:
                    weight_img = weight_img.transpose(Image.FLIP_LEFT_RIGHT)
                if flip[1,i]:
                    weight_img = weight_img.transpose(Image.FLIP_TOP_BOTTOM)
                weight_img = np.array(weight_img) / 255
<<<<<<< Updated upstream
                weight_img = (weight_img * 5 + 1 + y_arr_s) / 3
=======
                weight_img = weight_img * 9 + 1
                print(np.max(weight_img), np.min(weight_img))
>>>>>>> Stashed changes
                weight_img = np.expand_dims(weight_img, axis=2)

                X_d['input_weight'][i, ] = weight_img


            with Image.open(os.path.join(self.path, sample, 'smashing_border', '{}.png'.format(sample))) as y_img:
                y_img = y_img.resize(self.dim)
                if zoom_o[i]:
                    y_img = y_img.crop((zoom_o[i][0], zoom_o[i][1], zoom_o[i][2], zoom_o[i][3]))
                    y_img = y_img.resize(self.dim)
                y_img = y_img.rotate(rot[i])
                if flip[0,i]:
                    y_img = y_img.transpose(Image.FLIP_LEFT_RIGHT)
                if flip[1,i]:
                    y_img = y_img.transpose(Image.FLIP_TOP_BOTTOM)
                y_arr = np.array(y_img) / 255
                y_arr = np.expand_dims(y_arr, axis=2)
                y_arr[y_arr>0] = 1
                y_arr_sub = y_arr_store - y_arr
                y_arr_sub[y_arr_sub<0] = 0
                print(np.max(y_arr_sub), np.min(y_arr_sub))
                print(" ")
                Y[i, ] = y_arr_sub


        return X_d, Y


class PredictDataGenerator(DataGenerator):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, dim=(256,256), n_channels=1, mirror_edges=False):
        'Initialization'
        self.dim = dim
        self.batch_size = 8
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = False
        self.path = path
        self.mirror_edges = mirror_edges
        self.on_epoch_end()
        self.zoom = False

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index:(index+1)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.mirror_edges:
            out_shape = (self.dim[0] + self.mirror_edges, self.dim[0] + self.mirror_edges)
        else:
            out_shape = self.dim

        X = np.empty((self.batch_size, *out_shape, self.n_channels))

        with Image.open(os.path.join(self.path, list_IDs_temp[0], 'images', '{}.png'.format(list_IDs_temp[0]))) as x_img:
            x_img = x_img.resize(self.dim)
            x_img = x_img.convert(mode='L')
            x_img = ImageOps.autocontrast(x_img)
            x_img = x_img.rotate(270)

            for i in range(4):
                x_img = x_img.rotate(90)
                x_arr = np.array(x_img) / 255
                # x_arr = affine_transform(x_arr, [1,1], offset=[self.mirror_edges/2, self.mirror_edges/2],
                #                          output_shape=out_shape, mode='mirror')
                x_arr = np.expand_dims(x_arr, axis=2)
                X[i,] = x_arr

            x_img = x_img.rotate(90)
            x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
            x_img = x_img.rotate(270)

            for i in range(4):
                x_img = x_img.rotate(90)
                x_arr = np.array(x_img) / 255
                # x_arr = affine_transform(x_arr, [1,1], offset=[self.mirror_edges/2, self.mirror_edges/2],
                #                          output_shape=out_shape, mode='mirror')
                x_arr = np.expand_dims(x_arr, axis=2)
                X[i+4,] = x_arr

        return X




class PredictDataGenerator2(DataGenerator):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, dim=(256,256), n_channels=1, mirror_edges=False):
        'Initialization'
        self.dim = dim
        self.batch_size = 8
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = False
        self.path = path
        self.mirror_edges = mirror_edges
        self.on_epoch_end()
        self.zoom = False

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index:(index+1)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.mirror_edges:
            out_shape = (self.dim[0] + self.mirror_edges, self.dim[0] + self.mirror_edges)
        else:
            out_shape = self.dim

        X_input = np.empty((self.batch_size, *out_shape, self.n_channels))
        X_weigh = np.ones((self.batch_size, *out_shape, self.n_channels))


        X_d = {'input_image': X_input, 'input_weight': X_weigh}


        with Image.open(os.path.join(self.path, list_IDs_temp[0], 'images', '{}.png'.format(list_IDs_temp[0]))) as x_img:
            x_img = x_img.resize(self.dim)
            x_img = x_img.convert(mode='L')
            x_img = ImageOps.autocontrast(x_img)
            x_img = x_img.rotate(270)

            for i in range(4):
                x_img = x_img.rotate(90)
                x_arr = np.array(x_img) / 255
                # x_arr = affine_transform(x_arr, [1,1], offset=[self.mirror_edges/2, self.mirror_edges/2],
                #                          output_shape=out_shape, mode='mirror')
                x_arr = np.expand_dims(x_arr, axis=2)
                X_d['input_image'][i,] = x_arr

            x_img = x_img.rotate(90)
            x_img = x_img.transpose(Image.FLIP_LEFT_RIGHT)
            x_img = x_img.rotate(270)

            for i in range(4):
                x_img = x_img.rotate(90)
                x_arr = np.array(x_img) / 255
                # x_arr = affine_transform(x_arr, [1,1], offset=[self.mirror_edges/2, self.mirror_edges/2],
                #                          output_shape=out_shape, mode='mirror')
                x_arr = np.expand_dims(x_arr, axis=2)
                X_d['input_image'][i+4,] = x_arr



        return X_d



def post_process_predictions(arrays, transform=False):
    Y = np.zeros((1, *arrays[0].shape))
    for i in range(4):
        _arr = arrays[i]
        _arr = np.rot90(_arr, k=-1*i)
        if transform:
            _arr = _arr * 5 - 4
        Y = np.add(Y, _arr)

    for i in range(4):
        _arr = arrays[i+4]
        _arr = np.fliplr(_arr)
        _arr = np.rot90(_arr, k=i)
        if transform:
            _arr = _arr * 5 - 4
        Y = np.add(Y, _arr)

    return Y


def postprocess_list(ids, list_arrays):
    out_dict = dict.fromkeys(ids)
    for i, label in enumerate(ids):
        out_mask = post_process_predictions(list_arrays[0][(8 * i):(8 * i + 8)]) / 8
        out_bord = post_process_predictions(list_arrays[1][(8 * i):(8 * i + 8)]) / 8
        out_ = out_mask - out_bord > .01
        out_dict[label] = out_[0,:,:,0]
    return out_dict



def post_process_concat(ids, prediction, threshold=4, bool=True):
    prediction_for_ids = dict.fromkeys(ids)
    for i, label in enumerate(ids):
        if bool:
            d4_array = post_process_predictions(prediction[(8*i):(8*i+8)])  > threshold
        else:
            d4_array = post_process_predictions(prediction[(8*i):(8*i+8)]) / 8
        prediction_for_ids[label] = d4_array[0,:,:,0]
    return prediction_for_ids





def post_process_original_size(prediction_dict, path):
    org_size_prediction_for_ids = dict.fromkeys([ids for ids in prediction_dict.keys()])
    for label, pred in prediction_dict.items():
        with Image.open(os.path.join(path, label, 'images', '{}.png'.format(label))) as x_img:
            out_shape = np.array(x_img).shape
            pred_as_int = np.array(pred * 255, dtype=int)
            pred_as_  = imresize(pred_as_int,(out_shape[0],out_shape[1])) > 135.5
            # pred_as_int = np.array(pred, dtype=int)
            # pred_as_ = imresize(pred_as_int, (out_shape[0], out_shape[1])) > (255 / 2)
            pred_as_ = np.array(pred_as_, dtype=int)
            pred_as_ = morphology.remove_small_holes(pred_as_)
            pred_as_ = np.array(pred_as_, dtype=int)
            pred_as_ = morphology.remove_small_objects(pred_as_, 256)
            # pred_as_ = np.array(pred_as_, dtype=int)
            # pred_as_ = morphology.opening(pred_as_, morphology.square(3))
            pred_as_ = np.array(pred_as_, dtype=int)
            open_img = morph.binary_opening(pred_as_,iterations=1)
            # close_img = morph.binary_closing(pred_as_, iterations=1)
            # print(np.max(close_img))
            pred_as_ = open_img
            org_size_prediction_for_ids[label] = pred_as_

    return org_size_prediction_for_ids


def plot_image_true_mask(label, out, path):
    fig = plt.figure()
    with Image.open(os.path.join(path, label, 'images', '{}.png'.format(label))) as x_img:
        x_plot = x_img.convert(mode='L')
        x_arr = np.array(x_img)
        plt.subplot(131)
        plt.imshow(x_arr)

    if os._exists(os.path.join(path, label, 'mask', '{}.png'.format(label))):
        with Image.open(os.path.join(path, label, 'mask', '{}.png'.format(label))) as y_img:
            # y_arr = np.array(y_img)
            y_plot = y_img
            plt.subplot(132)
            plt.imshow(y_plot)
    else:
        print('no mask')

    out_arr = out * 255

    plt.subplot(133)
    plt.imshow(out_arr, cmap=cm.gray)


    fig.savefig('output_{}.png'.format(label))
    plt.close()


def plot_image_mask_border(label, out_mask, out_border, path):
    fig = plt.figure()
    with Image.open(os.path.join(path, label, 'images', '{}.png'.format(label))) as x_img:
        x_plot = x_img.convert(mode='L')
        # x_img = ImageOps.autocontrast(x_img)
        x_arr = np.array(x_img)
        plt.subplot(131)
        plt.imshow(x_arr)

    out_mask_p = out_mask * 255

    plt.subplot(132)
    plt.imshow(out_mask_p, cmap=cm.gray)

    out_border_p = out_border * 255

    plt.subplot(133)
    plt.imshow(out_border_p, cmap=cm.gray)

    # out_tot_p = out_tot
    # plt.subplot(144)
    # plt.imshow(out_tot_p, cmap=cm.gray)

    fig.savefig('C:/Users/huubh/Documents/DSB2018/output_{}.png'.format(label))
    plt.close()


def plot_image_mask_hyper_out(label, out_mask, out_mask_true, path, suffix=None, label_rgb=False, gt=False):
    fig = plt.figure(figsize=(16,9), dpi=150)
    ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=3)
    ax2 = plt.subplot2grid((3, 5), (0, 2), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((3, 5), (1, 2), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((3, 5), (2, 2), colspan=1, rowspan=1)
    ax5 = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=3)


    with Image.open(os.path.join(path, label, 'images', '{}.png'.format(label))) as x_img:
        x_plot = x_img.convert(mode='L')
        # x_img = ImageOps.autocontrast(x_img)
        x_arr = np.array(x_img)
        save_shape = x_arr.shape
        # plt.subplot(221)
        ax1.imshow(x_arr)
        ax1.axis('off')
        ax1.set_title('Input Image')

    with Image.open(os.path.join(path, label, 'mask', '{}.png'.format(label))) as x_img:
        x_plot = x_img.convert(mode='L')
        # x_img = ImageOps.autocontrast(x_img)
        x_arr = np.array(x_img)
        # ax2.subplot(222)
        ax2.imshow(x_arr, cmap=cm.gray)
        ax2.axis('off')
        ax2.set_title('Mask (ground truth)')

    with Image.open(os.path.join(path, label, 'smashing_border', '{}.png'.format(label))) as x_img:
        x_plot = x_img.convert(mode='L')
        # x_img = ImageOps.autocontrast(x_img)
        x_arr = np.array(x_img)
        # plt.subplot(223)
        ax3.imshow(x_arr)
        ax3.axis('off')
        ax3.set_title('Borders')

    out_mask_p = out_mask * 255

    # label_image = morphology.label(out_mask_p)
    # image_label_overlay = label2rgb(label_image, image=out_mask_p)

    # plt.subplot(224)
    ax4.imshow(out_mask_p, cmap=cm.gray, vmin=0, vmax=255)
    ax4.axis('off')
    ax4.set_title('Output')

    # print(np.max(out_mask_true), np.min(out_mask_true))
    if label_rgb:
        out_mask_p = out_mask_true * 255

        label_image = morphology.label(out_mask_p, background=0)
        # image_label_overlay = label2rgb(label_image, image=out_mask_p)
        out_mask_true = label_image
        out_mask_true = np.ma.masked_equal(out_mask_true, 0)

        import copy
        cmap = copy.copy(cm.prism)
        cmap.set_bad(color='black')

        ax5.imshow(out_mask_true, cmap=cmap)
        ax5.axis('off')
        ax5.set_title('Output (original size)')

    else:
        true_bord = x_arr
        true_bord[true_bord>0] = 1
        true_bord = morphology.erosion(true_bord)
        true_bord = imresize(true_bord, size=(save_shape[0], save_shape[1]))

        true_bord = np.ma.masked_equal(true_bord, 0)

        ax5.imshow(out_mask_true, cmap=cm.gray)
        if gt:
            ax5.imshow(true_bord, cmap=cm.prism, alpha=.95)
        ax5.axis('off')
        ax5.set_title('Output (original size)')

    fig.savefig('testt/{0}img{1}.png'.format(label[:5], suffix),bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    training = os.listdir('img')[0:8]
    training_generator = DataGenerator(training, 'img')
