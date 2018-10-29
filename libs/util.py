from random import randint
import itertools
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import sys
import os

xml_dir = '/home/ubuntu/Desktop/labelled_masks_ad/'
sys.path.append(os.path.join(os.path.join(os.path.dirname(sys.path[0]))))
def random_mask(height, width, channels=3):

    """Generates a random irregular mask with lines, circles and elipses"""    
    img = np.zeros((height, width, channels), np.uint8)
    
    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")    
    # Draw random lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
        
    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
        
    # Draw random ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)

    #Draw box to mask
    #print (filename.split('/')[-1])
    xml = (filename.split('/')[-1]).split('.')[0] + '.xml'
    xml_file = xml_dir + xml
    print (xml_file)
    tree = ET.parse(xml_file)
    xmin, xmax, ymin, ymax = '','','',''
    root = tree.getroot()
    bndbox = root[6][4]
    #print (bndbox.tag)
    xmin = int(bndbox[0].text)
    ymin = int(bndbox[1].text)
    xmax = int(bndbox[2].text)
    ymax = int(bndbox[3].text)
    #print (xmin, xmax, ymin, ymax)
    return 1-img

def custom_mask(height, width, channels=3):

    """Generates a random irregular mask with lines, circles and elipses"""    
    img = np.zeros((height, width, channels), np.uint8)
    coords = open('./coords.txt') 
    lines = coords.readlines()

    for coords in lines[1:]:
        coords = coords.split(' ')
        coords = [int(i.strip()) for i in coords]

        cv2.rectangle(img, (coords[0], coords[1]),(coords[3], coords[2]), (1,1,1), cv2.FILLED)




    return 1-img

custom_mask(512,512,3)
