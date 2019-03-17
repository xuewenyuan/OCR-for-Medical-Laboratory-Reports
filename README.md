# OCR-for-Medical-Laboratory-Reports

Test code for text detection and recognition from images of medical laboratory reports. The results may have some deviation on different devices.

The text detection module is forked from [endernewton](https://github.com/endernewton/tf-faster-rcnn). We improve the results through a patch-based strategy.
The text recognition module is implemented according to [meijieru](https://github.com/meijieru/crnn.pytorch). A concatenation structure is designed to utilize both the shallow and deep features, which results in a higher accuracy.

## Text detection
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
3. Download **the model** from [Google Drive](https://drive.google.com/file/d/1vIsscaZwqY0QWTu12Ah24gyYKuu7YF0y/view?usp=sharing).
4. Extract the .zip file and put them under "./detection/output/". Then the folder should be:
   ```
   ├── detection
    ├── output
      ├── vgg16
        ├── voc_2007_test
        ├── voc_2007_trainval
   ```
5. This framwork has been tested under Tensorflow 1.0. Update your -arch in setup script to match your GPU.
   ```
   cd detection/lib
   # Change the GPU architecture (-arch) if necessary
   vim setup.py
   # TitanX (Maxwell/Pascal)	sm_52
   # GTX 960M	sm_50
   # GTX 1080 (Ti)	sm_61
   # Tesla K80 (AWS p2.xlarge)	sm_37
   ```
6. Modify the GPU number before test.
   ```
   cd detection/tools
   vim printResults_with_crop.py
   
   33 os.environ['CUDA_VISIBLE_DEVICES']='1'
   ```
   Then, run "printResults_with_crop.py" under Tensorflow environment:
   ```
   cd ..
   python ./tools/printResults_with_crop.py --net vgg16 --dataset pascal_voc
   ```
   A result file will be created as "detection/tools/results.txt" 
7. Evaluaiton the results:
   ```
   python ./tools/eval_results.py
   ```
## Text Recognition 
1. Download **val set** from [Google Drive](https://drive.google.com/file/d/13zBDjnC88b5YmYxrDuvJrhcDDUOfnahE/view?usp=sharing). 
2. The test set has been transformed as LMDB format. Extract the .zip file and put them under "./recognition/data/val". Then the folder should be:
   ```
   ├── recognition
    ├── data
      ├── val
        ├── data.mdb
        ├── lock.mdb
   ```
3. Download **the model** from [Google Drive](https://drive.google.com/file/d/1o4UX7lO04NOncPpr8ecnTPOMAVCnDtJi/view?usp=sharing).
4. Put "netCRNN_L3.pth" under "./recognition/output/". Then the folder should be:
   ```
   ├── recognition
    ├── output
      ├── netCRNN_L3.pth
   ```
5. Dependence: Pytorch 0.2 and [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding).
6. Modify the GPU number before test.
   ```
   cd recognition
   vim eval.py
   
   16 os.environ['CUDA_VISIBLE_DEVICES']='1'
   ```
   Then, run "eval.py":
   ```
   python eval.py
   ```
