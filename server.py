import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import time
import numpy as np
import cv2 as cv
from flaskext.mysql import MySQL


import tensorflow as tf
import gc
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Flatten, Dense, LSTM, TimeDistributed, GlobalAveragePooling2D, MaxPooling2D
#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications import MobileNetV2
from keras.applications import DenseNet201
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



######################################################################

def gen_extractor(model_name):
    if model_name == 'vgg16_gavg':
        # Load vgg16 model
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

        # Freeze vgg16 layers
        for layer in vgg16.layers:
            layer.trainable = False

        # Extract features using vgg16
        inputs = Input(shape=(40, 112, 112, 3))
        l1 = TimeDistributed(vgg16)(inputs)
        l2 = TimeDistributed(GlobalAveragePooling2D())(l1)
        #outputs = TimeDistributed(Flatten())(l2)

        # Create model
        model = Model(inputs, l2)
    
    elif model_name == 'dens201_gavg':
        # Load ResNet50 model
        dens201 = DenseNet201(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

        # Freeze ResNet50 layers
        for layer in dens201.layers:
            layer.trainable = False

        # Extract features using ResNet50
        inputs = Input(shape=(40, 112, 112, 3))
        l1 = TimeDistributed(dens201)(inputs)
        l2 = TimeDistributed(GlobalAveragePooling2D())(l1)

        # Create model
        model = Model(inputs, l2)

    elif model_name == 'mobV2_gavg':
        # Load vgg16 model
        mobv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

        # Freeze vgg16 layers
        for layer in mobv2.layers:
            layer.trainable = False

        # Extract features using vgg16
        inputs = Input(shape=(40, 112, 112, 3))
        l1 = TimeDistributed(mobv2)(inputs)
        l2 = TimeDistributed(GlobalAveragePooling2D())(l1)

        # Create model
        model = Model(inputs, l2)

    else:
        print('Model not supported')

    return model

######################################################################

def gen_predictor(archi_p, weight_p):
    j_file = open(archi_p,'r')
    model_j = j_file.read()
    j_file.close()
    model = model_from_json(model_j)
    model.load_weights(weight_p)
    print("model loaded")

    return model


######################################################################

def preprocess_data(frame_file):
    
    #print("Loading frame", str(frame_file))
    #img = image.load_img(frame, target_size=(112, 112))
    #x = image.img_to_array(img)
    #print('Image type: ', type(frame))
    #print('Shape of image: ', frame.shape)
    #frame = np.resize(frame,(112,112,3)) --- np.resize destroy the image. suitable use is cv.resize
    #frame = frame/255
    #return np.expand_dims(frame, axis=0)

    frames = []
    for frame in range(40):
        img_path = os.path.join(frame_file, str(str(frame+1) + '.jpg'))
        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = x/255
        x = np.expand_dims(x, axis=0)
        frames.append(x)

    return frames

######################################################################

def extraction(frames, extractor):
    X = np.expand_dims(np.concatenate(frames), axis=0)
    X = extractor.predict_on_batch(X)
    return X

######################################################################

def prediction_temp(frames, predictor, class_list):
    
    X = frames
    print('Concat input shape: ', X.shape)
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
    print(class_list[7],': ',predicted[0,7].numpy())

    return result


######################################################################

LSTM_archi_path = './model_param/8_class/adam_16_dens201_gavg_8class.json'
LSTM_weight_path = './model_param/8_class/adam_16_dens201_gavg_8class.h5'
frames = []
prev_result = 10

print('loading lstm')
tf.keras.backend.clear_session()
predictor = gen_predictor(LSTM_archi_path, LSTM_weight_path)
print(predictor.summary())
print('loading extractor')
extractor = gen_extractor('dens201_gavg')
print(extractor.summary())


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Specify the upload folder
app.config['MYSQL_DATABASE_HOST'] = 'db'  # The hostname of your MySQL container
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'yeelight'
mysql = MySQL()
mysql.init_app(app)

######################################################################



######################################################################

def get_light_config(result):

    conn = mysql.connect()  # Get a connection to the MySQL database
    cursor = conn.cursor()

    # Retrieve data from the database for rendering in the template
    query = "SELECT value FROM light_value WHERE action_type LIKE %s"
    cursor.execute(query, (result))
    print("fetching result")
    setting = cursor.fetchall()[0][0]
    print('setting received: ' + str(setting))

    conn.close()  # Close the connection
    return setting 




######################################################################

#@app.route('/static/<path:filename>')
#def serve_static(filename):
#    root_dir = os.path.dirname(os.getcwd())  # Adjust this path to your project structure
#    return send_from_directory(os.path.join(root_dir, 'static'), filename)


@app.route('/configure', methods=['GET', 'POST'])
def configure():
    conn = mysql.connect()  # Get a connection to the MySQL database
    cursor = conn.cursor()
    
    if request.method == 'POST':
        action_type = request.form.get('action-type')
        temperature = request.form.get('temperature')

        # Update the database with the new numeric value
        query = "UPDATE light_value SET value = %s WHERE action_type = %s"
        cursor.execute(query, (temperature, action_type))
        conn.commit()

    # Retrieve data from the database for rendering in the template
    query = "SELECT action_type, value FROM light_value"
    cursor.execute(query)
    action_values = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()  # Close the connection

    return render_template('config.html', action_values=action_values)





@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File uploaded successfully'}), 200
    

    
@app.route('/batch_received', methods=['GET'])
def batch_received():
    global expected
    global frames
    global predictor
    global extractor
    global prev_result
    class_list = ['dining', 'drawing', 'mopping_floor', 'reading_book', 'running_on_treadmill','sleeping','watching_tv','yoga']
    frame_file = './uploads/'
    setting = 4004

    print('preprocessing')
    frames = preprocess_data(frame_file)
    print('extracting')
    X = extraction(frames, extractor)
    print('predicting')
    result = prediction_temp(X, predictor, class_list)
    if result != prev_result:
        print("new action detected")
        setting = get_light_config(class_list[result])
        prev_result = result

    time.sleep(5)
    return jsonify({'message': 'Batch of images received ' + str(result) + ' ' + str(setting), 'result': str(result), 'setting': str(setting)}), 200
    


######################################################################




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)