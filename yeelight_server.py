import requests
import numpy as np
import cv2 as cv
import gc
import time
from yeelight import discover_bulbs
from yeelight import Bulb



server_url = 'http://192.168.100.177:5000/upload'  # Update with your server URL
batch_size = 40


def create_yeelight():
    bulb_info = discover_bulbs()
    if len(bulb_info) > 0:
        print('Bulb detected')
        bulb = Bulb(bulb_info[0]['ip'])
        
        return True, bulb
    else:
        print('No bulb discovered')
        return False, None
    

def light_adjustment(bulb, result, setting):
    if bulb is None:
        print('No IoT light detected for configuration')
        return 0

    print('Activity ', str(result), ' detected. Start configure light')  
    if result != 5:
        bulb.set_brightness(100)
        bulb.set_color_temp(setting)
    elif result == 5:
        bulb.set_brightness(1)
        bulb.set_color_temp(setting)
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
    prev_result = 10
    class_list = ['dining', 'drawing','mopping floor', 'reading book','running on treadmill', 'sleeping', 'watching tv']
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
            i = 1
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
                    
                    img = cv.resize(img,(112,112))
                    _, img_encoded = cv.imencode('.jpg', img)
                    image_data = img_encoded.tobytes()
                    
                    files = {'file': (str(i) + '.jpg', image_data, 'image/jpeg')}
                    response = requests.post(server_url, files=files)
                    
                    if response.status_code == 200:
                        print("Image " + str(i) + " uploaded successfully.")
                        i += 1
                    else:
                        print("Error uploading image:" + str(i), response.json())


                    if i == batch_size + 1:
                        response = requests.get(server_url.replace("upload", "batch_received"))
    
                        if response.status_code == 200:
                            data = response.json()
                            result = data['result']
                            setting = data['setting']
                            if result != prev_result:
                                print('New action ' + class_list[int(result)] + ' detected')
                                if discovered:
                                    light_adjustment(yeelight, int(result), int(setting))
                                prev_result = result
                            print("Batch of images received. \n" + 'Result: ' + class_list[int(result)] + '\nSetting: ' + setting)
                            i = 1
                        else:
                            print("Error confirming batch received:", response.json())
                            i = 1
                        time.sleep(1)  # Add a delay to avoid overwhelming the server
    
                    
        
        
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


