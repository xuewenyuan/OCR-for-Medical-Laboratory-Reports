import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters() 
        recurrent, _ = self.rnn(input)

        #T, b, h = recurrent.size()
        b, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(b, T, -1)

        return output

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.conv0 = nn.Conv2d(1, 64, 3, 1, 1)
        self.relu0 = nn.ReLU(True)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.pool0_ = nn.AvgPool2d(4, 4)
        self.conv1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool1_ = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)#192
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(True)
        self.pool2_ = nn.AvgPool2d((2, 2), (2, 1), (0, 1))
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)#384
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.pool3_ = nn.AvgPool2d((2, 2), (2, 1), (0, 1))
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)#512
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(True)
        self.pool4 = nn.AvgPool2d((2, 2), (2, 1), (0, 1))
        self.conv5 = nn.Conv2d(768, 512, 3, 1, 1)
        self.relu5 = nn.ReLU(True)
        self.pool5 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv6 = nn.Conv2d(1024, 512, 2, 1, 0)
        self.relu6 = nn.ReLU(True)
        
	
        #self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
        #self.bn4 = nn.BatchNorm2d(512)
        #self.relu4 = nn.ReLU(True)
        #self.conv5 = nn.Conv2d(512, 512, 3, 1, 1)
        #self.relu5 = nn.ReLU(True)
        #self.pool5 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        #self.conv6 = nn.Conv2d(512, 512, 2, 1, 0)
        #self.relu6 = nn.ReLU(True)
        #self.conv_ = nn.Conv2d(512, 512, (4, 1), 1, 0)
        #self.relu_ = nn.ReLU(True)
        #self.convrd = nn.Conv2d(1024, 512, 1, 1, 0)
        #self.relurd = nn.ReLU(True)

        # cnn = nn.Sequential()
        #
        # def convRelu(i, batchNormalization=False):
        #     nIn = nc if i == 0 else nm[i - 1]
        #     nOut = nm[i]
        #     cnn.add_module('conv{0}'.format(i),
        #                    nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        #     if leakyRelu:
        #         cnn.add_module('relu{0}'.format(i),
        #                        nn.LeakyReLU(0.2, inplace=True))
        #     else:
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        #
        # convRelu(0)
        # cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        # convRelu(1)
        # cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        # convRelu(2, True)
        # convRelu(3)
        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        # convRelu(4, True)
        # convRelu(5)
        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        # convRelu(6, True)  # 512x1x16
        #
        # self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        #print(input.size())
        out0 = self.conv0(input)
        out0 = self.relu0(out0) #32x800
        out0_ = out0
        out0 = self.pool0(out0)

        out1 = self.conv1(out0) #16x400
        out1 = self.relu1(out1)
        out1_ = out1
        out1 = self.pool1(out1) #8x200
 
        #out0_ = self.pool0_(out0_) #8x200
        #cat0_1 = torch.cat((out1, out0_), dim=1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2) #8x200

        #out1_ = self.pool1_(out1_) #8x200
        #cat1_2 = torch.cat((out1_, out2), dim=1)

        out3 = self.conv3(out2) #8x200
        out3 = self.relu3(out3)
        out3_ = out3
        out3 = self.pool3(out3) #4x201

        #out2_ = self.pool2_(out2) #4x201
        #cat2_3 = torch.cat((out3, out2_), dim=1)

        out4 = self.conv4(out3)
        out4 = self.bn4(out4)
        out4 = self.relu4(out4) #4x201

        out3_ = self.pool3_(out3_) #4x201
        cat3_4 = torch.cat((out3_, out4), dim=1)

        out5 = self.conv5(cat3_4)
        out5 = self.relu5(out5)
        out5 = self.pool5(out5) #2x202

        out4 = self.pool4(out4) #2x202
        cat4_5 = torch.cat((out5, out4), dim=1)

        out6 = self.conv6(cat4_5)
        res = self.relu6(out6) #512*1*201

        b, c, h, w = res.size()
        assert h == 1, "the height of conv must be 1"
        res = res.squeeze(2)
        #res = res.permute(2, 0, 1)  # [w, b, c]
        res = res.permute(0, 2, 1)  # [b, w, c]
        output = self.rnn(res)

        return output
