import pandas as pd
import numpy as np
import os
import cv2
import xml.etree.ElementTree as et
from glob import glob
from keras.utils.image_utils import load_img, img_to_array

from PIL import Image, ImageDraw, ImageFont
from IPython.display import display

import easyocr
from sources.constant import *

ocr = easyocr.Reader(['en'])

def CreateListOfXmlfile(xml_folder_name):
  xml_path = os.path.join(os.getcwd(), xml_folder_name, '*xml')
  path = glob(xml_path)
  return path

'''
<annotation>
  <folder>images3</folder>
  <filename>SNDR001.jpg</filename>
  <path>I:\workspace\project_neuron\vscode\asrec\images3\SNDR001.jpg</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>4000</width>
    <height>3000</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>roadnames</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>1773</xmin>
      <ymin>322</ymin>
      <xmax>3104</xmax>
      <ymax>561</ymax>
    </bndbox>
  </object>
</annotation>
'''

def CreateListOfBBox(xml_filepath):
    target_label = dict(xmlfile=[], xmin=[], xmax=[], ymin=[], ymax=[])
    for xml_file in xml_filepath:
        # Read in the file with ElementTree
        tree = et.parse(xml_file)
        # Get the root element of the XML file
        # Use the root to access and manipulate data in the XML file
        root = tree.getroot()
        # From root get the object
        object = root.find('object')
        # From the object get the root
        bb = object.find('bndbox')
        # Get the bounding box coordinates
        xmin = int(bb.find('xmin').text)
        xmax = int(bb.find('xmax').text)
        ymin = int(bb.find('ymin').text)
        ymax = int(bb.find('ymax').text)
        # Appending extracted coordinates into a dictionary
        target_label['xmlfile'].append(xml_file)
        target_label['xmin'].append(xmin)
        target_label['xmax'].append(xmax)
        target_label['ymin'].append(ymin)
        target_label['ymax'].append(ymax)
    df = pd.DataFrame(target_label)
    #bb_df = list()
    bb_df = df[['xmin', 'xmax', 'ymin', 'ymax']].values
    #df.to_csv(label_filename, index= False)
    return bb_df

def GetImageFilePath(folder_name, filename):
    # Get the file name from the XML file
    image_name = et.parse(filename).getroot().find('filename').text
    # Get the path to the file name
    image_path = os.path.join(os.getcwd(), folder_name, image_name)
    return image_path

def CreateListOfImages(xml_files):
  image_names = list()
  for i in xml_files:
    image_names.append(GetImageFilePath(TRAIN_DIR, i))
  return image_names

def NormalizeImage(image_names, labels):
  #d = dict()
  norm_labels = list()
  norm_features = list()

  for index in range(len(image_names)):
    ## 1. Load the original size images with Keras ##
    img_name = image_names[index]
    img_obj = load_img(img_name)

    ## 2. Normalize the lables ##
    # Step 1. Convert image objects to arrays
    img_obj_arr = img_to_array(img_obj)
    # Step 2. Get the original image size and depth to be used 
    # for normalizing process
    height, width, depth = img_obj_arr.shape
    # Step 3. Get the bounding box coordinates from the label dataframe
    xmin, xmax, ymin, ymax = labels[index]
    # Step 4. Normalize the labels
    xmin_norm, xmax_norm = xmin/width, xmax/width
    ymin_norm, ymax_norm = ymin/height, ymax/height
    label_norm = (xmin_norm, xmax_norm, ymin_norm, ymax_norm)
    # Step 5. Store the normalized labels to a list
    norm_labels.append(label_norm)

    ## 3. Normalize the original images ##
    # Step 1. Resize the original image sizes to 224x224
    img_obj_new = img_obj.resize([IMAGE_SIZE,IMAGE_SIZE])
    #print(img_obj_new)
    # Step 2. Convert them to arrays
    img_obj_new_arr = img_to_array(img_obj_new)
    # Step 3. Devide the image arrays with the maximum pixel values
    img_obj_new_arr_norm = img_obj_new_arr/255.0
    # 9. Append to data list
    norm_features.append(img_obj_new_arr_norm)

  # ## 4. Store the list of labels and features into a dictionary
  # d['labels'] = labels
  # d['features'] = features

  return norm_features, norm_labels

def PredictObjectLocation(test_path, prediction_path, model):
    '''
    The goal is to get the height, width, and depth from an original image.
    Those information are used to create denormalized parameters.
    '''
    # Load the original size images with Keras and convert it to
    # 8-bit array represented from 0 to 255
    img_orig_size = load_img(test_path)
    img_orig_size_arr = np.array(img_orig_size, dtype=np.uint8)
    # Get the height, width, and depth
    height, width, depth = img_orig_size_arr.shape
    denorm_constant = np.array([width, width, height, height])

    '''
    For the purpose of prediction, different sizes of images have to be
    converted to a same size. In this case, the size of 244x244 has been
    chosen. The images have been resized then go to a normalization process.
    However, before that process, an image has to be converted to an array
    format. Then, after that we can normilized all values by 255.0.
    '''
    # Load the image using Keras load_img and resize it at the same time
    img_pred_size = load_img(test_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    # Convert it into array and get the normalized output
    norm_img_pred_size_arr = np.array(img_pred_size, dtype=np.uint8)/255.0
    # The above result has to be converted into a dynamic fourth dimension
    # 1 indicates the number of image, As here we are passing only one image
    norm_test_arr = norm_img_pred_size_arr.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)

    '''
    Run predictions and get the result in terms of the coordinates of  the bounding box
    '''
    norm_pred_bb_coords = model.predict(norm_test_arr)
    
    '''
    The predicted coordinates are in the normalized version. Since we want 
    to draw the predicted bounding box coordinates on the unnormalized original
    image, we have to denormalized the bb coordinates using the denormalized
    parameters created from the original image.
    '''
    # Denormalize the bounding box coordinates
    unnorm_pred_bb_coords = (norm_pred_bb_coords * denorm_constant).astype(np.int32)
    
    '''
    Draw the bounding box on top of the image and save it using cv2 imwrite.
    '''
    # Draw the bounding box on top of the image
    xmin, xmax, ymin, ymax = unnorm_pred_bb_coords[0]
    below_left_point = (xmin, ymin)
    top_right_point = (xmax, ymax)
    cv2.rectangle(img_orig_size_arr, 
                  below_left_point, 
                  top_right_point, 
                  (0, 255, 0), 5)
    # Since the original image was read as RGB using Keras load_img,
    # it has to be converted to bgr before saving it using cv2 imwrite
    img_orig_size_arr_bgr = cv2.cvtColor(img_orig_size_arr, cv2.COLOR_RGB2BGR)
    # Save it
    cv2.imwrite(prediction_path, img_orig_size_arr_bgr)

    # Return the b coordinates
    return unnorm_pred_bb_coords


def PerformOcr(test_folder, prediction_folder, filename, model):
    '''
    1. Load the test image and convert it into the array format
    '''
    # Get the test image name with its path
    test_img_path = os.path.join(os.getcwd(), test_folder, filename)
    # Load the test image and convert it into the array format
    test_img_arr = np.array(load_img(test_img_path))

    '''
    2. Predict the bounding box coordinates
    '''
    print(f'The predicting image: {filename}')
    # Get the prediction image name with its absolute path
    prediction_path = os.path.join(os.getcwd(), prediction_folder, filename)
    cordinates = PredictObjectLocation(test_img_path, prediction_path, model)
    
    '''
    3. Display the prediction image
    '''
    # Read the image using PIL Image function
    prediction_img = Image.open(prediction_path)
    # Display it
    display(prediction_img)

    '''
    4. Get the image within the bounding box area and save it
    '''
    # Get the bonding box coordinates
    xmin, xmax, ymin, ymax = cordinates[0]
    # Get the rectangular area based on the coordinates
    rect = test_img_arr[ymin:ymax, xmin:xmax]
    #print(rect)
    # Convert colours from RGB to BGR
    rect_bgr = cv2.cvtColor(rect, cv2.COLOR_RGB2BGR)
    #gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) # Not being used
    # Save the bounding box area
    rect_filename = 'BB_' + filename
    rect_path = os.path.join(os.getcwd(), BBOX_DIR, rect_filename)
    cv2.imwrite(rect_path, rect_bgr)

    '''
    5. Perform OCR
    '''
    result = ocr.readtext(rect_path)

    '''
    6. Display the bounding box images with detected areas and detected texts
    '''
    # Load the image
    image = Image.open(rect_path).convert('RGB')
    # Create a draw object
    draw = ImageDraw.Draw(image)
    # Decide the font type and size
    font_path = os.path.join(os.getcwd(), 'sources/simfang.ttf')
    print(font_path)
    font = ImageFont.truetype(font_path, 18)
    # Draw the detected areas on the draw object
    y = 5
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        draw.line([*top_left, *top_right, *bottom_right, *bottom_left, *top_left], 
                  fill='red', 
                  width=6)
        text = "Text: %s, Probability: %f" % (text, prob)
        print(text)
        draw.text((5, y), text, fill ="blue", font = font, align ="left")
        y = y+25
    # Display the draw area
    display(image)