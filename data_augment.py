import numpy as np 
import cv2 
import os 
import matplotlib.image as mpimg
# import pandas as pd 
# from sklearn.model_selection import train_test_split

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 200, 66, 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

def load_image(data_dir, image_file):
    # return cv2.imread(os.path.join(data_dir, image_file))
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def crop(image):
    # cut the sky and 
    return image[60:-25, :, :]

def resize(image):
    return cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def choose_image(data_dir, left, center, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2 
    return load_image(data_dir, center), steering_angle

def random_flip(image, steering_angle):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x=100, range_y=10):
    
    random_width = range_x * (np.random.rand() - 0.5)
    random_height = range_y *  (np.random.rand() - 0.5)
    steering_angle += random_width*0.002
    T = np.float32([[1, 0, random_width], [1, 0, random_height]])
    width, height = image.shape[:2]
    image_translation = cv2.warpAffine(image, T, (width, height))

    return image_translation, steering_angle

def random_shadow(image):
    # (x1, y1) and (x2, y2) forms a line
    x1, y1 = IMG_WIDTH * np.random.rand(), 0
    x2, y2 = IMG_WIDTH * np.random.rand(), IMG_HEIGHT
    # (xm, ym) all (x, y) in image
    xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    # all of points below line set 1 and 0 otherwise
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0] = 1
    
    # random to set light == 1 or == 0
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.6)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio

    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) 

def random_brightness(image):

    # use HSV(Hue, Saturation, Value) is also called HSB(B for Brightness)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio 

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 

def augument(data_dir, center, left, right, steering_angle):

    image, steering_angle = choose_image(data_dir, left, center, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle)
    image = random_shadow(image)
    image = random_brightness(image)

    return image, steering_angle 

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    # initializing batch data 	
    images = np.empty([batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for idx in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[idx]
            steering_angle = steering_angles[idx]
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            images[i] = preprocess(image)
            steers[i] = steering_angle 
            i += 1
            if i == batch_size:
                break
        yield images, steers 


# data_dir = 'data'
# data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'driving_log.csv'), 
#                     names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

# X = data_df[['center', 'left', 'right']].values
# y = data_df['steering'].values
# pos_zero = np.array(np.where(y==0)).reshape(-1, 1)

# pos_none_zero = np.array(np.where(y!=0)).reshape(-1, 1)
# np.random.shuffle(pos_zero)
# pos_zero = pos_zero[:1000]
# # join two numpy arrays
# pos_combined = np.vstack((pos_zero, pos_none_zero))
# pos_combined = list(pos_combined)
# X = X[pos_combined].reshape(-1, 3)
# y = y[pos_combined].reshape(-1)
# batch_size = 32
# X_train, y_train, X_valid, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
# batch_generator(data_dir, X_train, y_train, batch_size, True)
