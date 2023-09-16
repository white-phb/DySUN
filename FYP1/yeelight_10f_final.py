import numpy as np
import cv2 as cv
from yeelight import discover_bulbs
from yeelight import Bulb

import tensorflow as tf
import gc
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Flatten, Dense, LSTM, TimeDistributed, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils import plot_model
import tensorflow.keras.utils as image

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
            
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus),"Physical GPUs, ",len(logical_gpus),"Logical GPUs")
    except RuntimeError as e:
        print(e)

def gen_extractor(model_name):
    if model_name == 'vgg16_gavg':
        # Load vgg16 model
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

        # Freeze vgg16 layers
        for layer in vgg16.layers:
            layer.trainable = False

        # Extract features using vgg16
        inputs = Input(shape=(10, 112, 112, 3))
        l1 = TimeDistributed(vgg16)(inputs)
        l2 = TimeDistributed(GlobalAveragePooling2D())(l1)
        #outputs = TimeDistributed(Flatten())(l2)

        # Create model
        model = Model(inputs, l2)
    
    elif model_name == 'resnet_gavg':
        # Load ResNet50 model
        resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

        # Freeze ResNet50 layers
        for layer in resnet.layers:
            layer.trainable = False

        # Extract features using ResNet50
        inputs = Input(shape=(20, 112, 112, 3))
        l1 = TimeDistributed(resnet)(inputs)
        l2 = TimeDistributed(GlobalAveragePooling2D())(l1)

        # Create model
        model = Model(inputs, l2)

    elif model_name == 'vgg16_avg':
        # Load vgg16 model
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

        # Freeze vgg16 layers
        for layer in vgg16.layers:
            layer.trainable = False

        # Extract features using vgg16
        inputs = Input(shape=(20, 112, 112, 3))
        l1 = TimeDistributed(vgg16)(inputs)
        l2 = TimeDistributed(AveragePooling2D(pool_size=(2,2),strides=1))(l1)
        outputs = TimeDistributed(Flatten())(l2)

        # Create model
        model = Model(inputs, outputs)

    else:
        print('Model not supported')

    return model

def gen_predictor(archi_p, weight_p):
    j_file = open(archi_p,'r')
    model_j = j_file.read()
    j_file.close()
    model = model_from_json(model_j)
    model.load_weights(weight_p)
    print("model loaded")

    return model


def preprocess_data(frame):
    
    #print("Loading frame", str(frame_file))
    #img = image.load_img(frame, target_size=(112, 112))
    #x = image.img_to_array(img)
    #print('Image type: ', type(frame))
    #print('Shape of image: ', frame.shape)
    #frame = np.resize(frame,(112,112,3))
    frame = frame/255
    return np.expand_dims(frame, axis=0)

def extraction(frames, extractor):
    X = np.expand_dims(np.concatenate(frames), axis=0)
    X = extractor.predict_on_batch(X)
    return X


def prediction(frames, extractor, predictor, class_list):
    
    X = np.expand_dims(np.concatenate(frames),axis=0)
    X = extractor.predict_on_batch(X)

    predicted = predictor.predict_on_batch(X)
    result = np.argmax(predicted, axis=1)
    result_label = class_list[result]
    
    return result, result_label

def prediction_temp(frames, extractor, predictor, class_list):
    
    X = np.expand_dims(frames,axis=0)
    print('Concat input shape: ', X.shape)
    #print('Start extract feature')
    #X = extractor.predict(X)
    print('Start prediction feature')
    predicted = predictor(X)
    result = np.argmax(predicted, axis=1)[0]
    result_label = class_list[result]

    print('Predicted result: ', result_label)
    print(class_list[0],': ',predicted[0,0].numpy())
    print(class_list[1],': ',predicted[0,1].numpy())
    print(class_list[2],': ',predicted[0,2].numpy())
    print(class_list[3],': ',predicted[0,3].numpy())
    print(class_list[4],': ',predicted[0,4].numpy())
    print(class_list[5],': ',predicted[0,5].numpy())
    print(class_list[6],': ',predicted[0,6].numpy())

    return result


def create_yeelight():
    bulb_info = discover_bulbs()
    if len(bulb_info) > 0:
        print('Bulb detected')
        bulb = Bulb(bulb_info['ip'])
        
        return True, bulb
    else:
        print('No bulb discovered')
        return False, None
    

def light_adjustment(bulb, result):
    if bulb is None:
        print('No IoT light detected for configuration')
        return 0

    print('Activity ', str(result), ' detected. Start configure light')  
    if result == 0:
        bulb.set_brightness(100)
        bulb.set_color_temp(3000)
    elif result == 1:
        bulb.set_brightness(100)
        bulb.set_color_temp(5000)
    elif result == 2:
        bulb.set_brightness(100)
        bulb.set_color_temp(6500)
    elif result == 3:
        bulb.set_brightness(100)
        bulb.set_color_temp(4500)
    elif result == 4:
        bulb.set_brightness(100)
        bulb.set_color_temp(5000)
    elif result == 5:
        bulb.set_brightness(100)
        bulb.set_color_temp(1700)
    elif result == 6:
        bulb.set_brightness(100)
        bulb.set_color_temp(3000)
    else:
        print('Activity to be supported in the future')


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def main():
    # ---------------------------------- SET CONSTANT
    LSTM_archi_path = '/home/jeheng/Downloads/CSI-Camera-master/vgg16_gavg/model.json'
    LSTM_weight_path = '/home/jeheng/Downloads/CSI-Camera-master/vgg16_gavg/model.h5'
    class_list = ['dining', 'drawing', 'mopping_floor', 'reading_book', 'running_on_treadmill','sleeping','watching_tv']
    frames = []
    frame_length = 40
    print('loading lstm')
    tf.keras.backend.clear_session()
    predictor = gen_predictor(LSTM_archi_path, LSTM_weight_path)
    #print(predictor.summary())
    print('loading extractor')
    extractor = gen_extractor('vgg16_gavg')
    #print(extractor.summary())
    p_frames = []

    for chance in range(5):
        discovered, yeelight = create_yeelight()
        if not discovered:
            print('Error in discovering yeelight bulb')
        else:
            print('Light discovered')
            break


    video_capture = cv.VideoCapture(gstreamer_pipeline(flip_method=0), cv.CAP_GSTREAMER)
    window_title = "CSI Camera"

    if video_capture.isOpened():

        try:
            window_handle = cv.namedWindow(window_title, cv.WINDOW_AUTOSIZE)
            while True:
                ret_val, img = video_capture.read()
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv.getWindowProperty(window_title, cv.WND_PROP_AUTOSIZE) >= 0:
                    cv.imshow(window_title, img)
                    if not ret_val:
                        print('fail to stream')
                        break

                    if len(p_frames) != frame_length:
                        if len(frames) < 10:
                            print('loading begin')
                            #frame = preprocess_data(frame)
                            frame = preprocess_data(cv.resize(img,(112,112)))
                            frames.append(frame)
                        else:
                            if len(p_frames) != 0:
                                print('extracting 10 frames')
                                p_frames = np.concatenate((np.squeeze(p_frames),np.squeeze(extraction(frames, extractor))))
                            else:
                                print('extracting first 10 frames')
                                p_frames = extraction(frames, extractor)

                            frames.clear()
                            gc.collect()
            			
                    else:
                        print('Prediction begin')
                        print('Shape of input: ', p_frames.shape)
                        result = prediction_temp(p_frames, extractor, predictor, class_list)
                        light_adjustment(yeelight, result)
                        p_frames = []
                        gc.collect()


                else:
                    break 
                keyCode = cv.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv.destroyAllWindows()

    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    main()






