from imutils import paths
from keras.models import load_model
from data import load_data
from keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EPOCHS = 40
INIT_LR = 1e-4
BS = 8
CLASS_NUM = 4
norm_size = 448
trainset_size = 8047
test_1 = 'image/test_1'
test_2 = 'image/test_2'
train = 'image/train'
validation = 'image/validation'
pathes = [test_2, test_1, validation, train]
'''
trainX, trainY = load_data(train_file_path, norm_size, CLASS_NUM)
validationX, validationY = load_data(validation_file_path, norm_size, CLASS_NUM)
test_1X, test_1Y = load_data(test_1_file_path, norm_size, CLASS_NUM)
test_2X, test_2Y = load_data(test_2_file_path, norm_size, CLASS_NUM)
'''
modelpath = 'model/checkpoint_densenet121448adam-70e-val_acc_0.65.hdf5'
#modelpath = 'model\model1.hdf5'
model = load_model(modelpath)



def geti(l):
    maxx = 0
    index = 0
    for i in range(4):
        if l[i] > maxx:
            maxx = l[i]
            index = i
    return index

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=classes)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


lis1 = ['1', '2', '3', '4']
lis2 = [i for i in range(4)]
dic = dict(zip(lis2, lis1))

for j in range(len(pathes)):
    y_true = []
    y_pre = []
    imagePaths = list(paths.list_images(pathes[j]))
    labels = [int(i.split(os.path.sep)[-2]) for i in imagePaths]
    for i in range(len(labels)):
        labels[i] = int(labels[i])
    for i in tqdm(range(len(imagePaths))):
        image = cv2.imread(imagePaths[i])
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = np.array(image, dtype="float") / 255.0
        res = model.predict(image)[0]
        #print(res)
        #print(dic[geti(res)])
        y_pre.append(dic[geti(res)])
        y_true.append(dic[labels[i]])
    plot_confusion_matrix(y_true, y_pre, classes=lis1,
                      title=pathes[j].split('/')[-1])
    plt.savefig('image/confusionmatrix/{}_{}.png'.format(modelpath.split('/')[-1], pathes[j].split('/')[-1]))
'''
validationX, validationY = load_data(validation, norm_size, CLASS_NUM)
test_1X, test_1Y = load_data(test_1, norm_size, CLASS_NUM)

loss_and_acc_1 = model.evaluate(test_1X, test_1Y, verbose=1)

print('test_1_loss: {:.4f} - test_1_acc: {:.4f}'.format(loss_and_acc_1[0], loss_and_acc_1[1]))

loss_and_acc_2 = model.evaluate(validationX, validationY, verbose=1)
print('val_loss: {:.4f} - val_acc: {:.4f}'.format(loss_and_acc_2[0], loss_and_acc_2[1]))
'''
'''
y_true = []
y_pre = []
imagePaths = list(paths.list_images(test_2))
labels = [int(i.split(os.path.sep)[-2]) for i in imagePaths]
for i in range(len(labels)):
    labels[i] = int(labels[i])

for i in tqdm(range(len(imagePaths))):

    image = cv2.imread(imagePaths[i])
    image = cv2.resize(image, (norm_size, norm_size))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.array(image, dtype="float") / 255.0
    res = model.predict(image)[0]
    #print(res)
    #print(dic[geti(res)])
    y_pre.append(dic[geti(res)])
    y_true.append(dic[labels[i]])

plot_confusion_matrix(y_true, y_pre, classes=lis1,
                      title='Confusion matrix, without normalization')

plt.show()
'''