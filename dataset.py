import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf

from augment import *

def build_decoder(with_labels=True, target_size=(256, 256), mask_size=None):
    if mask_size is None:
        mask_size = target_size
    else:
        mask_size = (mask_size,mask_size)

    def decode_img_preprocess(img, sizes):
        img = tf.image.resize(img, sizes)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def decode_img(path, sizes=target_size, is_gray=False):
        """
        path to image
        """
        file_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
        img = tf.image.rgb_to_grayscale(img) if is_gray else img
        img = decode_img_preprocess(img, sizes)
        return img
        
    def decode_with_labels(path_img, path_mask):
        return decode_img(path_img, sizes=target_size, is_gray=False), decode_img(path_mask, sizes=mask_size, is_gray=True)
        
    return decode_with_labels if with_labels else decode_img

def do_multi_scale_image(img, target_size):
    # img_x2 = tf.image.resize(img, (target_size[0]//2,target_size[1]//2))
    # img_x4 = tf.image.resize(img, (target_size[0]//4,target_size[1]//4))
    # img_x8 = tf.image.resize(img, (target_size[0]//8,target_size[1]//8))
    # img_x16 = tf.image.resize(img, (target_size[0]//16,target_size[1]//16))
    # img_x32 = tf.image.resize(img, (target_size[0]//32,target_size[1]//32))
    # return {'x1':img, 'x2':img_x4, 'x3':img_x8, 'x4':img_x16, 'x5':img_x32}
    img_x4 = tf.image.resize(img, (target_size[0]//4,target_size[1]//4))
    return img_x4

def build_dataset(paths, labels=None, bsize=32,
                  decode_fn=None, augment=None, batch_augment_img=None,
                  repeat=False, shuffle=1024,
                  cache=False, cache_dir="", 
                  img_size=(256,256), multi_scale_output=False):
    """
    paths: paths to images
    labels: int label
    """              
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    AUTO = tf.data.experimental.AUTOTUNE
    dataset_input = tf.data.Dataset.from_tensor_slices((paths))
    dataset_label = tf.data.Dataset.from_tensor_slices((labels))

    dset = tf.data.Dataset.zip((dataset_input, dataset_label))
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.map(augment, num_parallel_calls=AUTO) if augment is not None else dset
    
    if batch_augment_img is not None:
        dset = dset.batch(bsize)
        dset = dset.map(batch_augment_img, num_parallel_calls=AUTO)
        dset = dset.unbatch()
        dset = dset.map(lambda x,y:(clip_image(x),clip_image(y)), num_parallel_calls=AUTO)

    dset = dset.map(lambda x,y:(x,do_multi_scale_image(y,img_size)), num_parallel_calls=AUTO) if multi_scale_output else dset
    dset = dset.batch(bsize)
    dset = dset.prefetch(AUTO)
    
    return dset

def build_dataset_from_X_Y(X_path, Y_int, with_labels, img_size,
                           batch_size, repeat, shuffle, augment, batch_augment, multi_scale_output, mask_size=None):
    decoder = build_decoder(with_labels, img_size, mask_size)

    augment_img = build_augment() if augment else None
    batch_augment_img = build_batch_augment() if batch_augment else None

    dataset = build_dataset(X_path, Y_int, bsize=batch_size, decode_fn=decoder,
                            repeat=repeat, shuffle=shuffle, augment=augment_img, batch_augment_img=batch_augment_img,
                            img_size=img_size, multi_scale_output=multi_scale_output)

    return dataset

if __name__ == '__main__':
    from utils import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)

    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)

    train_img_paths = glob('unzip/polyp/TrainDataset/images/*')
    train_mask_paths = glob('unzip/polyp/TrainDataset/masks/*')

    valid_img_paths = []
    valid_mask_paths = []

    valid_route = 'unzip/polyp/TestDataset'
    for valid_data in os.listdir(valid_route):
        valid_path = path_join(valid_route, valid_data)
        valid_img_dir = path_join(valid_path, 'images')
        valid_mask_dir = path_join(valid_path, 'masks')

        valid_img_paths += glob(valid_img_dir + '/*')
        valid_mask_paths += glob(valid_mask_dir + '/*')

    print(len(train_img_paths))
    print(len(valid_img_paths))
    print(train_img_paths[0])
    print(train_mask_paths[0])
    print(valid_img_paths[0])
    print(valid_mask_paths[0])

    # train_augment = False
    mask_size = None

    train_n_images = len(train_img_paths)
    train_dataset = build_dataset_from_X_Y(train_img_paths, train_mask_paths, train_with_labels, img_size,
                                           BATCH_SIZE, train_repeat, train_shuffle, train_augment, train_batch_augment, train_multi_scale_output)

    valid_n_images = len(valid_img_paths)
    valid_dataset = build_dataset_from_X_Y(valid_img_paths, valid_mask_paths, valid_with_labels, img_size,
                                           VALID_BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment, valid_batch_augment, valid_multi_scale_output)

    for x, y in train_dataset:
        break
    print(x)
    print(y)

    import cv2
    import numpy as np

    print("min x:", np.min(x))
    print("max x:", np.max(x))
    print("min y:", np.min(y))
    print("max y:", np.max(y))

    cv2.imwrite("sample_dataset_img.png", np.array(x[0][...,::-1])*255)

    if not train_multi_scale_output:
        cv2.imwrite("sample_dataset_mask.png", np.array(y[0][...,::-1])*255)
    else:
        print(type(y))
        print(y.keys())
        for idx, (key, value) in enumerate(y.items()):
            value = value[0][...,::-1] * 255
            print(value.shape)
            cv2.imwrite(f"sample_dataset_masks_multi_scale_{key}.png", np.array(value))

