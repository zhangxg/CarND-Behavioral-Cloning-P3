import csv
from glob import glob

import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Flatten, Lambda, Cropping2D, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# load csv
def load_and_split_data(data_dir, test_size=0.2, file_names=None, load_three=False, steer_left=0.08, steer_right=0.04):
    """
    load the training image path and steer angle from the csv file. 
    my traing data is collected incrementally, each time a new folder is generated, which contains the driving_log.csv
    and IMG folder, the new folder is put into the 'data' folder, which is entrance point to all the training images, 
    below is the structure:
    
    root_folder:
        --models.py
        --data
            --0314
                --driving_log.csv
                --[IMG]
            --0315
                --driving_log.csv
                --[IMG]

    Arguments:
        test_size: used to split the train and validation set
        
        file_names: the director holding the dirving_log.csv
        
        load_three: if the left and rigth camero images be loaded
        
        steer_left: if the load_three is true, then this value is used to adjust the steer angle for left images
        
        steer_right: same as steer_left, but applied to the right images. 
    
    """
    
    files = []
    if file_names == None:  # if not assined, then using all the training data
        files = glob(data_dir + "*/*.csv")
    else:
        files = [(data_dir + f + "/driving_log.csv") for f in file_names]
    samples = []
    for f in files:
        with open(f) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                steer_center = float(line[3])
                if load_three:
                    samples.append(["." + line[0][line[0].find("/data/"):], steer_center])
                    samples.append(["." + line[1][line[1].find("/data/"):], steer_center + steer_left])
                    samples.append(["." + line[2][line[2].find("/data/"):], steer_center - steer_right])
                else:
                    samples.append(["." + line[0][line[0].find("/data/"):], steer_center])
    if test_size:
        train_samples, validation_samples = train_test_split(samples, test_size=test_size)
        return train_samples, validation_samples
    else:
        return samples, []


def generator(samples, batch_size=32, with_shuffle=True):
    """
    data generator.
    Arguments:
        batch_size: the number of samples yield each time
        
        with_shuffle: shuffle the data of not, the shuffle is applied to the whole images first, 
            and then to the batched samples before yielding. 
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        if with_shuffle:
            shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for sample in batch_samples:
                images.append(cv2.imread(sample[0]))
                angles.append(sample[1])
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train) if with_shuffle else (X_train, y_train)
            
            
def Nivdia(dropout_ratio=0.5):
    """
    the CNN model. 
    
    the idea is from this paper: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    
    the architecture mentioned in this paper is tailored. 
    
    """
    model = Sequential()
    model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    
    model.add(Convolution2D(3, 5, 5, border_mode="valid", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(dropout_ratio))

    model.add(Convolution2D(24, 5, 5, border_mode="valid", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(dropout_ratio))

    model.add(Convolution2D(36, 3, 3, border_mode="valid", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(dropout_ratio))

    model.add(Convolution2D(16, 3, 3, border_mode="valid", activation="relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(dropout_ratio))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation("relu"))

    model.add(Dense(84))
    model.add(Activation("relu"))

    model.add(Dense(1))
    return model
    
    
# the training procedure
model = Nivdia(dropout_ratio=0.25)

train_samples, validation_samples = load_and_split_data('./data/', 
                                                        file_names=["0314", "0319-2"],
                                                        load_three=True,
                                                        steer_left=0.04,
                                                        steer_right=0.04
                                                       )

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model.compile(loss='mse', optimizer="adam")
checkpointer = ModelCheckpoint(filepath="./weights.{epoch:02d}-{val_loss:.2f}.h5", verbose=1, save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

history_object = model.fit_generator(train_generator, samples_per_epoch= \
                len(train_samples), validation_data=validation_generator, 
                nb_val_samples=len(validation_samples), nb_epoch=20, callbacks=[checkpointer, earlyStopping])

model.save("model.h5")
