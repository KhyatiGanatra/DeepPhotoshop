# Deep Photoshop 
<b>Deep Photoshop</b> is a tool that helps users to fill missing patches in the image.    
It is currently focussed on removing company logos.

Having ads/company logos in an image is not always desirable.   
Also, it will restrict the user to share images publicly as the company can claim copyrights for the same.   

This project is motivated by this idea to remove ads in images automatically using Deep Learning technique.   

Google slides for the project can be found [here]

Installation Guide:
Cloning the repo gives you the object detection and infilling model.

After using it on a sample image with ad, following steps are to be followed:
1. Detect the logo in the image
2. Mask the logo
3. Use infilling model to generate newly filled image
