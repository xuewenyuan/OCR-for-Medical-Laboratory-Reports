import torch
from warpctc_pytorch import CTCLoss
from torch.autograd import Variable
import utils
import dataset
import keys
import os
from os.path import join as osj
import sys
from scipy import misc
from timer import Timer
from PIL import Image

import models.crnn_fromL4 as crnn

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
reload(sys)
sys.setdefaultencoding('utf-8')


def minDistance(w1, w2):
    w1 = w1.encode('utf-8')
    w2 = w2.encode('utf-8')
    m=len(w1)+1
    n=len(w2)+1

    dp = [[0 for i in range(n)] for j in range(m)]#(m+1)*(n+1)

    for i in range(n):
        dp[0][i]=i

    for i in range(m):
        dp[i][0]=i

    for i in range(1,m):

        for j in range(1,n):

            dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1, dp[i-1][j-1]+(0 if w1[i-1]==w2[j-1] else 1))

    return dp[m-1][n-1]

def getFileList(path):
    ret = []
    folders = []
    for rt,dirs,files in os.walk(path):
        #for filename in files:
        #    ret.append(filename)
        for folder in dirs:
		    filePath = osj(path,folder)
		    for _rt,_dirs,_files in os.walk(filePath):
		        for filename in _files:
		            ret.append(osj(filePath,filename))
    return ret

model_path = '/data/xuewenyuan/crnn-pytorch-model/expr_fromL4/netCRNN_5_5000.pth'
#model_path = "/data/xuewenyuan/crnn-pytorch-model/expr_crnn/netCRNN_5_45000.pth"
#val_path = './data/srcFile/val'
val_path = '/home/zoe/PycharmProjects/testdataset/roi'
#alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
#val_fileList = getFileList(val_path)

#ConvertToUTF8(val_fileList)
'''
model = crnn.CRNN(32, 1, 352, 256)
if torch.cuda.is_available():
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(1))
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(keys.alphabet)

transformer = dataset.resizeNormalize((800, 32))

correct = 0.0
count = 0.0

for filename in os.listdir(val_path):
    filepath = os.path.join(val_path,filename)
    ext_name = os.path.splitext(filepath)
    if ext_name[1] =='.JPG':
        count += 1

for i in range(1,int(count)+1):
    img_name = (val_path + '/%d.JPG')%i
    image = Image.open(img_name).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    text_name = (val_path + '/%d.txt')%i
    with open(text_name,'r') as f:
        text = f.read()
        text = unicode(text, 'utf8')
        text = text.replace(' ','')
        #fin = open('.'+f.split('.')[1]+'.dat','r')
        #gt = fin.readline()
    if text==sim_pred:
        correct+=1
    #print('%-20s => %-20s'% (raw_pred, sim_pred))
    #print('gt: '+gt)
print("accuracy: %f count: %d" % (correct/count, count))
'''
#opt_valroot = "/home/xuewenyuan/crnn.pytorch/data/lmdb/val"
opt_valroot = "/home/xuewenyuan/crnn.pytorch/data/cmdd-multi-val/val"
opt_batchSize = 20
opt_workers = 2

test_dataset = dataset.lmdbDataset(
    root=opt_valroot, transform=dataset.resizeNormalize((800, 32)))
#test_dataset = dataset.lmdbDataset(root=opt_valroot)
print(len(test_dataset))

converter = utils.strLabelConverter(keys.alphabet)
criterion = CTCLoss()

crnn = crnn.CRNN(32, 1, 352, 256)
if torch.cuda.is_available():
    crnn = crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(1))
print('loading pretrained model from %s' % model_path)
crnn.load_state_dict(torch.load(model_path))

image = torch.FloatTensor(opt_batchSize, 3, 32, 32)
text = torch.IntTensor(opt_batchSize * 5)
length = torch.IntTensor(opt_batchSize)
image = Variable(image)
text = Variable(text)
length = Variable(length)

print('Start val')

for p in crnn.parameters():
    p.requires_grad = False

crnn.eval()
data_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=opt_batchSize,
    num_workers=int(opt_workers), drop_last=False)

#data_loader = torch.utils.data.DataLoader(
#    test_dataset, shuffle=False, batch_size=opt_batchSize,
#    num_workers=int(opt_workers), drop_last=False,
#    collate_fn=dataset.alignCollate(imgH=32, imgW=100, keep_ratio=True))


val_iter = iter(data_loader)

i = 0
n_correct = 0
n_minDistance = 0
count = 0
loss_avg = utils.averager()

_t = Timer()

max_iter = len(data_loader)
err_ind = 0
statistic = []
for i in range(max_iter):
    if i%100 == 0:
        print(str(i)+'/'+str(max_iter))
    data = val_iter.next()
    i += 1
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    count += batch_size
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    _t.tic()
    preds = crnn(image)
    _t.toc()
    preds_size = Variable(torch.IntTensor([preds.size(1)] * batch_size))
    preds = preds.permute(1,0,2)
    cost = criterion(preds, text, preds_size, length) / batch_size
    loss_avg.add(cost)

    _, preds = preds.max(2)
    #preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    for pred, target, img in zip(sim_preds, cpu_texts, image):
        if pred == target:
            n_correct += 1
            statistic.append(1)
        else:
            err_ind += 1
            statistic.append(0)
            n_minDistance += minDistance(pred,target)
            #img = image.data.cpu().numpy()
            #path = './data/wrong_re1-name-10w/img_{}_{}.jpg'.format(err_ind,pred.replace('/','+'))
            #misc.imsave(path, img[0][0])

#with open('./McNemar/crnn.txt','a') as f:
#    for st in statistic:
#        f.write(str(st)+'\n')

#raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
#for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
    #print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

#accuracy = n_correct / float(max_iter * opt.batchSize)
accuracy = n_correct / float(count)
avg_minDistance = n_minDistance/float(count-n_correct)
print('Test loss: %f accuray: %f AvgMinDistance: %f samples: %d' % (loss_avg.val(), accuracy, avg_minDistance, count))
print('{:.3f}s/batch, {:.3f}s/image'.format(_t.average_time, _t.total_time/count))
