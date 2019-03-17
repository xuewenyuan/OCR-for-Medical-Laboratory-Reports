import matplotlib.pyplot as plt

with open("/home/xuewenyuan/crnn.pytorch/log/fromL4/crnn.txt.2018-07-21_00-06-04",'r') as f:
    texts = f.readlines()

train_loss = []
train_iter = []
test_loss = []
test_iter = []
test_accuracy = []

n = 1000
for line in texts:
    if line[0] == '[':
        line = line.split(' ')
        train_loss.append(line[2])
        train_iter.append(n)
        n += 1000
    if line[0] == 'T':
        line = line.split(' ')
        #loss.append(line[2])
        test_loss.append(line[2])
        test_accuracy.append(line[4])
        line[0] = n
        test_iter.append(n)


colors = 'navy'
#plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(train_iter, train_loss, 'b', lw=2, label='train loss')
ax1.plot(test_iter, test_loss, 'r', lw=2, label='test loss')
ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax1.legend(loc='upper left')
ax1.set_ylim(0, 5)
ax2 = ax1.twinx() # this is the important function
ax2.plot(test_iter, test_accuracy, 'g',label = 'test accuracy')
ax2.legend(loc='upper right')
ax2.set_ylabel('accuracy');
plt.show()
