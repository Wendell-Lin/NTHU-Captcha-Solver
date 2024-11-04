import tflite_runtime.interpreter as tflite
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sys
import os
from PIL import Image
import requests
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warnings and info

# A utility function to decode the output of the network
def decode(preds):
    np.argmax(preds, axis=2)
    p = np.argmax(preds, axis=2)
    to_num = ['?','4', '5', '0', '6', '3', '7', '9', '1', '8', '2']
    ans = ''
    for idx, i in enumerate(p[0]):
        if i == 11:
            continue
        # fix adjacent problem
        elif idx + 1 < len(p[0]) and i == p[0][idx + 1]:
                continue
        else:
            ans += to_num[i]

    return ans

def load_image(url_and_cookie):
    url, cookie = url_and_cookie
    response = requests.get(url, cookies={'PHPSESSID':cookie})
    img = Image.open(BytesIO(response.content)).convert('L')
    np_img = np.array(img)
    np_img = (np_img / 255.).astype(np.float32)
    np_img = np.expand_dims(np_img, axis=-1)
    np_img = np_img.transpose(1, 0, 2)
    np_img = np.expand_dims(np_img, axis=0)
    return np_img

def load_images(url, PHPSESSID):
    url_and_cookie = [(url, PHPSESSID)] * 5
    pool = ThreadPoolExecutor(max_workers=5)

    images = []

    for image in pool.map(load_image, url_and_cookie):
        images.append(image)

    return images
        
def predict(images):
    interpreter = tflite.Interpreter(model_path="./model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Structuring because TFlite doesn't do that
    tflite_results = []
    for i in images:
        interpreter.set_tensor(input_details[0]['index'], i)

        interpreter.invoke()

        tflite_result = interpreter.get_tensor(output_details[0]['index'])
        tflite_results.append(decode(tflite_result))


    del interpreter
    print(tflite_results)
    return max(tflite_results,key=tflite_results.count) 

def main(argv):
    img = load_images(argv[1], argv[2])
    preds = predict(img)
    print(preds)
    
main(sys.argv)