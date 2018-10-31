git clone https://github.com/pjreddie/darknet
cd darknet
make
cp ../yolo_custom_files/image.c ./src/image.c
cp ../yolo_custom_files/yolov2_logo_detection.cfg ./cfg/

