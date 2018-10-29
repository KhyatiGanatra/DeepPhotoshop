# Deep Photoshop 
<b>Deep Photoshop</b> is an image infilling model that currently focusses on removing company logos.

## Motivation
Having ads/company logos in an image is not always desirable. Also, it will restrict the user to share images publicly as the company can claim copyrights for the same. We could try to remove the logos by simply cropping the image. But if the ad is in the center of an image, this approach fails. So, hereby I created a pipeline that can remove the logo automatically from the image and replace that part with the context of the image.    
The project is thus motivated by this idea to remove ads in images using Deep Learning technique.   

Google slides for the project can be found [here](https://docs.google.com/presentation/d/1sWU5M_oBRgBjet9fjsCapiLf3tbw1_htFnHM6wIR26U/edit?usp=sharing)

## Overview of the project:   
The project aims to remove the logo from the image. This happens in two major steps:     
   1. Logo Detection and then mask the logo
         Logo detection is a YOLO model trained on Flickr 47 dataset.
   2. Infill the masked part
         Model trained on infilling random shaped mask in dataset of building images will fill the masked part
    
## Results:    
Here are the results of the pipeline:    
The image on the left is the input image with company logo/ad. Then we have two images which after masking the logo and the one after infilling.    


<img src='images/3.jpg' width='250' height='250'/><img src='images/3_op.png' width='600' height='270'/>

Let's see step wise output of the two models:

### YOLO:
Here, we identify the company logo and draw bounding box around it. We just need the co-ordinates of the box and label is not important. So need not worry about label at this point.    
<img src='./images/Google-4.jpg' width='312' height='312'/>

### Image Infilling:
This takes the masked image as an input. As an output it gives an image after filling the gap.    
<img src='./images/4_op.png' width='612' height='300'/>

### Few more results:
<img src='images/1.jpg' width='250' height='250'/><img src='images/1_op.png' width='600' height='270'/> 
<img src='images/5.jpg' width='250' height='250'/><img src='images/5_op.png' width='600' height='270'/>

## Installation Guide:

CLone this repo.    
You also need to download the weights for Object detection and image infilling.
Both can be found [here](https://drive.google.com/drive/folders/1r7PEIqbsgZBY42kW_yIpm8Jk1hbQ8POr?usp=sharing)
You can put these weights in the yolo_custom_files directory.    

For using it on a sample image with ad (you can use one of the images from images directory if needed), following steps are to be followed:    
0. Put the image in test_infilling/test and convert image in required dimensions.    
   python src/processing.py ./data/test_infilling/test/image_name    
1. Detect the logo in the image     
   cd darknet    
   ./darknet detect ../yolo_custom_files/yolov2_logo_detection.cfg ../yolo_custom_files/YOLOv2_logo_detection_10000th_iteration.weights ../data/test_infilling/test/image_name 
   cd ..            
   
3. Use infilling model to generate newly filled image   
   python src/predict.py  
   
You should be able to see the images in custom_results folder.
     
This will generate output images in the data/custom_results folder   
