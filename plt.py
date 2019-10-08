import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

file0 = open('log/log_train.txt', 'rt')
file1 = open('log1/log_train.txt', 'rt')
file2 = open('log2/log_train.txt', 'rt')
file3 = open('log3/log_train.txt', 'rt')
y_file0 = [[], [], []]
y_file1 = [[], [], []]
y_file2 = [[], [], []]
y_file3 = [[], [], []]
for line in file0.readlines():
    if 'eval mean loss:' in line:
        str = line.strip().replace('eval mean loss:', '').replace(' ', '')
        y_file0[0].append(float(str))
    if 'eval accuracy:' in line:
        str = line.strip().replace('eval accuracy:', '').replace(' ', '')
        y_file0[1].append(float(str))
    if 'eval avg class acc: ' in line:
        str = line.strip().replace('eval avg class acc: ', '').replace(' ', '')
        y_file0[2].append(float(str))

for line in file1.readlines():
    if 'eval mean loss:' in line:
        str = line.strip().replace('eval mean loss:', '').replace(' ', '')
        y_file1[0].append(float(str))
    if 'eval accuracy:' in line:
        str = line.strip().replace('eval accuracy:', '').replace(' ', '')
        y_file1[1].append(float(str))
    if 'eval avg class acc: ' in line:
        str = line.strip().replace('eval avg class acc: ', '').replace(' ', '')
        y_file1[2].append(float(str))

for line in file2.readlines():
    if 'eval mean loss:' in line:
        str = line.strip().replace('eval mean loss:', '').replace(' ', '')
        y_file2[0].append(float(str))
    if 'eval accuracy:' in line:
        str = line.strip().replace('eval accuracy:', '').replace(' ', '')
        y_file2[1].append(float(str))
    if 'eval avg class acc: ' in line:
        str = line.strip().replace('eval avg class acc: ', '').replace(' ', '')
        y_file2[2].append(float(str))

for line in file3.readlines():
    if 'eval mean loss:' in line:
        str = line.strip().replace('eval mean loss:', '').replace(' ', '')
        y_file3[0].append(float(str))
    if 'eval accuracy:' in line:
        str = line.strip().replace('eval accuracy:', '').replace(' ', '')
        y_file3[1].append(float(str))
    if 'eval avg class acc: ' in line:
        str = line.strip().replace('eval avg class acc: ', '').replace(' ', '')
        y_file3[2].append(float(str))
x = np.arange(250)
plt.figure(figsize=(30, 15))
plt.plot(x, y_file0[0], c='b')
plt.plot(x, y_file1[0], c='y')
plt.plot(x, y_file2[0], c='r')
plt.plot(x, y_file3[0], c='g')
plt.legend(['eval mean loss-f0', 'eval mean loss-f1', 'eval mean loss-f2', 'eval mean loss-f3'])
plt.show()
plt.figure(figsize=(30, 15))
plt.plot(x, y_file0[1], c='b')
plt.plot(x, y_file1[1], c='y')
plt.plot(x, y_file2[1], c='r')
plt.plot(x, y_file3[1], c='g')
plt.legend(['eval accuracy-f0', 'eval accuracy-f1', 'eval accuracy-f2', 'eval accuracy-f3'])
plt.show()
plt.figure(figsize=(30, 15))
plt.plot(x, y_file0[2], c='b')
plt.plot(x, y_file1[2], c='y')
plt.plot(x, y_file2[2], c='r')
plt.plot(x, y_file3[2], c='g')
plt.legend(['eval avg class acc-f0', 'eval avg class acc-f1', 'eval avg class acc-f2','eval avg class acc-f3'])
plt.show()

file_0 = open('dump/log_evaluate.txt', 'rt')
file_1 = open('dump1/log_evaluate.txt', 'rt')
file_2 = open('dump2/log_evaluate.txt', 'rt')
file_3 = open('dump3/log_evaluate.txt', 'rt')
for i in range(7):
    next(file_0)
    next(file_1)
    next(file_2)
    next(file_3)
y_file_0 = []
y_file_1 = []
y_file_2 = []
y_file_3 = []
label = []
for line in file_0.readlines():
    label.append(line.strip().split(':')[0].strip())
    y_file_0.append(float(line.strip().split(':')[1].strip()))
for line in file_1.readlines():
    y_file_1.append(float(line.strip().split(':')[1].strip()))
for line in file_2.readlines():
    y_file_2.append(float(line.strip().split(':')[1].strip()))
for line in file_3.readlines():
    y_file_3.append(float(line.strip().split(':')[1].strip()))
bar_width = 0.2
X = np.arange(40)
plt.figure(figsize=(30, 15))
plt.bar(X, y_file_0, bar_width, label='f0')
plt.bar(X + bar_width, y_file_1, bar_width, label='f1')
plt.bar(X + bar_width + bar_width, y_file_2, bar_width, label='f2')
plt.bar(X + bar_width + bar_width + bar_width, y_file_3, bar_width, label='f3')
plt.legend()
plt.xticks(X + bar_width * 2, label)
plt.show()
