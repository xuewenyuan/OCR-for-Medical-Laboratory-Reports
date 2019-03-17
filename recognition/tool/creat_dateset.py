import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from os.path import join as osj
import keys

def mycmp(content):
    return len(content['label_text'])

def getFileList(path):
    ret = []
    folders = []
    for rt,dirs,files in os.walk(path):
        #for filename in files:
        #    ret.append(osj(path,filename))
        for folder in dirs:
		    filePath = osj(path,folder)
		    for _rt,_dirs,_files in os.walk(filePath):
		        for filename in _files:
		            ret.append(osj(filePath,filename))
    return ret


def ConvertToUTF8(datFileList):
    # creat dict
    map_dict = {}
    for i, char in enumerate(keys.alphabet):
        map_dict[i] = char

    # read .txt files
    datFileList.sort()
    for f in datFileList:
        if f.split('.')[1] == 'txt':
            #print(f)
            txt_label = open(f, 'r').read()
            #txt_label = txt_file.read()
            #txt_file.close()
            txt_label = txt_label.split(',')
            txt_label.pop()
            result = ''
            for label in txt_label:
                result = result + (map_dict[int(label)])
            f_dat = open(f.split('.')[0]+'.dat', 'w')
            #print(f.split('.')[0]+'.dat')
            f_dat.write(result.encode('utf-8'))
    print('Convert to UTF8 done.')

def genImgLabel_List(fileList):
    ConvertToUTF8(fileList)
    ImageSet = {}
    LabelSet = {}
    nameList = []
    index = 0
    for n,f in enumerate(fileList):
        if n%10000==0:
           print('fetch file: %d %%' % (int(100*n/(1.0*len(fileList)))))
        fname = f.split('.')[0]
        
        #if fname not in nameList:
        #   index
        #   nameList.append(fname)
           #print('fetch file: '+fname)
        if f.split('.')[1] == 'jpg' or f.split('.')[1] == 'JPG':
            ImageSet[fname] = f
            nameList.append(fname)
        """
        if f.split('.')[2] == 'txt':
            labelFile = open(f, 'r')
            LabelSet[fname] = labelFile.read()
        """
        if f.split('.')[1] == 'dat':#txt
            labelFile = open(f, 'r')
            label = labelFile.read()
            #label = unicode(label, "utf-8")
            #label = unicode(label, 'utf8')
            label = label.replace(' ','')
            LabelSet[fname] = label
            #print(fname)

    print 'len of namelist',len(nameList)
    dataList = []
    for name in nameList:
        content = {}
        content['img_path'] = ImageSet[name]
        content['label_text'] = LabelSet[name]
        dataList.append(content)
    
    dataList.sort(key = mycmp)
    
    imgpathList = []
    labelList = []
    for data in dataList:
        imgpathList.append(data['img_path'])
        labelList.append(data['label_text'])
    return imgpathList, labelList

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    """
    test_root = './data/srcFile/test'
    test_fileList = getFileList(test_root)
    testImagelist, testLabelList = genImgLabel_List(test_fileList)
    """
    #train_root =  '/data/xuewenyuan/Dataset/realEstate/data/train' #'./data/srcFile/train'
    val_root = '/home/xuewenyuan/Dataset/CMDD/cmdd-multi-ocr-val2' #'./data/srcFile/val'

    #train_fileList = getFileList(train_root)
    val_fileList = getFileList(val_root)
    #print('search files done: %d, %d.' % (len(train_fileList), len(val_fileList)))
    
    #trainImageList, trainLabelList = genImgLabel_List(train_fileList)
    valImagelist, valLabelList = genImgLabel_List(val_fileList)
    print('generate image label done.')
    
    #createDataset('/home/xuewenyuan/crnn.pytorch/data/RE/train', trainImageList, trainLabelList)
    createDataset('/home/xuewenyuan/crnn.pytorch/data/cmdd-multi-val2/val', valImagelist, valLabelList)
