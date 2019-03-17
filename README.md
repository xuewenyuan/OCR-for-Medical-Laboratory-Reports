# OCR-for-Medical-Laboratory-Reports
Text Detection and Recognition from Images of Medical Laboratory Reports with Deep-Learning-Based Approach

The text detection module is forked from [endernewton](https://github.com/endernewton/tf-faster-rcnn)

## Test for Text detection on the CMDD dataset
1. Download **the dataset** from [Google Drive](https://drive.google.com/file/d/1c8F2ZmqFhvc8_QQEJBxKhUYdsMSBgReJ/view?usp=sharing). 
2. Extract the .zip file and put them under "./detection/data/VOCdevkit2007/". Then the folder should be:
   ```
   ├── detection
    ├── data
      ├── VOCdevkit2007
        ├── VOC2007
          ├── Annotations
          ├── ImageSets
          ├── JPEGImages
          ├── editAnnotation.py
          ├── labels_src.json
   ```
3. Download **the model** from [Google Drive]().
4. Extract the .zip file and put them under "./detection/output/". Then the folder should be:
   ```
   ├── detection
    ├── output
      ├── vgg16
        ├── voc_2007_test
        ├── voc_2007_trainval
   ```
