try:
    from keras_implementation import generator
    from keras_implementation import pipeline
except:
    import generator, pipeline
import keras.metrics
keras.metrics.mean_iou_border = pipeline.mean_iou_border
keras.losses.cust = keras.losses.binary_crossentropy
keras.metrics.mean_iou = pipeline.mean_iou
from keras import backend as K

from keras import models
import os
import numpy as np
import glob
import re
import pandas as pd

# def rle_encoding(x):
#
#     dots = np.where(x.T.flatten() == True)[0]
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if (b>prev+1): run_lengths.extend((b + 1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def out_dict_to_rle_dict(dict_with_arrays):
    rle_dict = dict.fromkeys([ids for ids in dict_with_arrays.keys()])
    for id, array_out in dict_with_arrays.items():
        rle = rle_encode(array_out)
        rle_dict[id] = rle
    return rle_dict


from skimage.morphology import label # label regions
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)


def one_at_a_time(path_models, path_img, labels):
    prediction_generator = generator.PredictDataGenerator(labels, path_img)
    for i in [999]:
        # with models.load_model(all_models[model_hd5]) as model_xx:
        model_xx = models.load_model('model_dogeCC12.h5')
        prediction_x8 = model_xx.predict_generator(prediction_generator)
        del model_xx
        K.clear_session()
        print(labels)
        prediction = generator.post_process_concat(labels, prediction_x8, threshold=4.25, bool=False)
        # predicti_t = generator.post_process_concat(labels, prediction_x8, threshold=4.25, bool=True)
        out_true = generator.post_process_original_size(prediction, path_img)
        for ids, out_arra in prediction.items():
            generator.plot_image_mask_hyper_out(ids, out_arra, out_true[ids], path_img, str(i))
        for ids, out_arra in prediction.items():
            generator.plot_image_mask_hyper_out(ids, out_arra, out_true[ids], path_img, 'rgb', True)
        for ids, out_arra in prediction.items():
            generator.plot_image_mask_hyper_out(ids, out_arra, out_true[ids], path_img, 'gt', False, True)

    all_models = glob.glob(os.path.join(path_models, '*.hd5'))
    model_sorted = np.argsort([int(re.search('model_x_(.*)[.]', model).group(1)) for model in all_models])
    for i, model_hd5 in enumerate(model_sorted):
        # if i > 110:
        print(all_models[model_hd5])
        # with models.load_model(all_models[model_hd5]) as model_xx:
        model_xx = models.load_model(all_models[model_hd5])
        prediction_x8 = model_xx.predict_generator(prediction_generator)
        del model_xx
        K.clear_session()
        print(labels)
        prediction = generator.post_process_concat(labels, prediction_x8, threshold=4.25, bool=False)
        # predicti_t = generator.post_process_concat(labels, prediction_x8, threshold=4.25, bool=True)
        out_true = generator.post_process_original_size(prediction, path_img)
        for ids, out_arra in prediction.items():
            generator.plot_image_mask_hyper_out(ids, out_arra, out_true[ids], path_img, "%03d" % i)
        # make video
        # ffmpeg -r 5 -i output_00ae6_%03d.png -vcodec mpeg4 -y movie.mp4


if __name__ == '__main__':
    path_img = 'C:/Users/huubh/Documents/DSB2018_bak/img'
    # path_img = '../stage2_test_final'
    labels = os.listdir(path_img)[:]
    print(len(labels))
    prediction_ids = labels[:]
    labels = ['449f41710769584b5e4eca8ecb4c76d5272605f27da2949e6285de0860d2cbc0',
              '8c3ef7aa7ed29b62a65b1c394d2b4a24aa3da25aebfdf3d29dbfc8ad1b08e95a',
              '4193474b2f1c72f735b13633b219d9cabdd43c21d9c2bb4dfc4809f104ba4c06',
              '853a4c67900c411abd04467f7bc7813d3c58a5f565c8b0807e13c6e6dea21344',
              '6b0ac2ab04c09dced54058ec504a4947f8ecd5727dfca7e0b3f69de71d0d31c7',
              '9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32',
              'fdda64c47361b0d1a146e5b7b48dc6b7de615ea80b31f01227a3b16469589528',
              '7f38885521586fc6011bef1314a9fb2aa1e4935bd581b2991e1d963395eab770']


    one_at_a_time(path_models='C:/Users/huubh/Dropbox/DSB_MODEL/show', path_img=path_img, labels=labels)
    # #
    # #
    quit(2)

    # model_x5 = models.load_model('C:/Users/huubh/Dropbox/DSB_MODEL/model_x88.h5')
    # model_b5 = models.load_model('C:/Users/huubh/Dropbox/DSB_MODEL/model_t5t.h5')
    model_b5 = models.load_model('model_dogeCC12.h5')


    prediction_generator_boundaries = generator.PredictDataGenerator(prediction_ids[:], path_img)
    predictions_boundaries = model_b5.predict_generator(prediction_generator_boundaries)

    # prediction_generator_only_masks = generator.PredictDataGenerator(prediction_ids[:], path_img)
    # predictions_only_masks = model_x5.predict_generator(prediction_generator_only_masks)
    #
    # henk_ = generator.post_process_concat(prediction_ids[:], predictions_only_masks, threshold=4, bool=True)

    nico_ = generator.post_process_concat(prediction_ids[:], predictions_boundaries, threshold=4.25, bool=True)

    # for ids, out_arra in nico_.items():
    #     generator.plot_image_mask_hyper_out(ids, out_arra, path_img)

    # out_true = generator.post_process_original_size(out_square, path_img)

    # summed_dict = dict.fromkeys([ids for ids in out_masks_square.keys()])
    #
    # for id in summed_dict.keys():
    #     summed_dict[id] = out_masks_square[id] - out_boundaries_square[id] > .5

    out_true = generator.post_process_original_size(nico_, path_img)

    # for ids, out_arra in nico_.items():
    #     generator.plot_image_mask_border(ids, out_arra, out_true[ids], path_img)

    # for ids, out_arra in nico_.items():
    #     generator.plot_image_mask_hyper_out(ids, out_arra, path_img)


    # for ids, out_arra in out_masks_square.items():
    #     summed_dict[ids] = generator.plot_image_mask_border(ids, out_masks_square[ids], out_arra, out_true[ids], path_img)

    # segmentation = morphology.watershed(elevation_map, markers)
    # segmentation_clean = np.array(segmentation < 1.5, dtype=int)
    # segmentation_clean = morphology.remove_small_objects(segmentation, 1)


    # for ids, out_arra in out_true.items():
    #     print(np.max(out_arra))
    #     generator.plot_image_true_mask(ids, out_arra, path_img)

    new_test_ids = []
    rles = []

    for id, arrayx_ in out_true.items():
        rle = list(prob_to_rles(arrayx_))
        rles.extend(rle)
        new_test_ids.extend([id] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub_45.csv', index=False)

    # for id, arrayx_ in out_true.items():
    #     rle = list(prob_to_rles(arrayx_))
    #     rles.extend(rle)
    #     new_test_ids.extend([id] * len(rle))
    #
    # sub = pd.DataFrame()
    # sub['ImageId'] = new_test_ids
    # sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    # sub.to_csv('sub.csv', index=False)






    ####

