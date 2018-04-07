try:
    from keras_implementation import generator
    from keras_implementation import pipeline
except:
    import generator, pipeline
import keras.metrics
keras.metrics.mean_iou = pipeline.mean_iou
from keras import models
import os
import numpy as np
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

if __name__ == '__main__':
    path_img = '../img'
    labels = os.listdir(path_img)[:]
    print(labels)
    prediction_ids = labels[:]

    # model_x5 = models.load_model('model_x88.h5')
    model_b5 = models.load_model('C:/Users/huubh/Dropbox/DSB_MODEL/model_e2e.h5')

    prediction_generator_boundaries = generator.PredictDataGenerator(prediction_ids[:], path_img)
    predictions_boundaries = model_b5.predict_generator(prediction_generator_boundaries)

    henk_ = generator.postprocess_list(prediction_ids[:], predictions_boundaries)

    nico_ = generator.post_process_concat(prediction_ids[:], predictions_boundaries[0], threshold=4, bool=True)

    for ids, out_arra in henk_.items():
        generator.plot_image_mask_border(ids, nico_[ids], out_arra, path_img)

    # out_true = generator.post_process_original_size(out_square, path_img)

    # summed_dict = dict.fromkeys([ids for ids in out_masks_square.keys()])
    #
    # for id in summed_dict.keys():
    #     summed_dict[id] = out_masks_square[id] - out_boundaries_square[id] > .5

    out_true = generator.post_process_original_size(nico_, path_img)

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
    sub.to_csv('sub.csv', index=False)






    #####

