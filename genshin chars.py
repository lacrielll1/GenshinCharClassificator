path = "g:/proj_data/train_cleared/"

from PIL import Image, ImageFile
from PIL import ImageFont
from PIL import ImageDraw 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim import Adam
from torch import nn
from torch.utils.data import random_split
from sklearn.metrics import classification_report
import time
from torchmetrics import Accuracy
import numpy as np
import torchvision.utils as vutils
import os
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as f
import timm

BS = 32
IMSIZE = 224
epochs = 15


def main():
    global device, d, num_classes
    dataset = datasets.ImageFolder(path, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))]))
    print(len(dataset))
    spl = int(0.8 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [spl, len(dataset) - spl])

    dataloader_train = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(val_set, batch_size=BS, shuffle=False, num_workers=4)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    d = dict([(i, val) for i, val in enumerate(dataset.classes)])
    print(d)
    
    num_classes = len(d)
    print("num classes: ", num_classes)
    dataiter = iter(dataloader_train)
    images, labels = next(dataiter)
    figure = plt.figure(figsize=(15, 15))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(images), size=(1,)).item()
        img, label = images[sample_idx], labels[sample_idx]
        img += torch.normal(img.max() / 100, img.max() / 100, img.size())
        img -= img.min()
        img /= img.max()
        img = img.permute(1, 2, 0)
        figure.add_subplot(rows, cols, i)
        plt.title(d[label.item()])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

    data = get_test_data('g:/proj_data/test/')
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    print("efficient_net: ", count_trainable_parameters(model), "params")
    #model.load_state_dict(torch.load("g:/proj_data/efficient_net_weights.pth"))
    h2 = fit(model, dataloader_train, dataloader_val, epochs=6, add_noise=True, model_name='efficient_net')
    p, pr = predict(model, data)


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def fit(model, train_data, test_data=None, epochs=2, add_noise=False, model_name='model', val_iter=5):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    num_batches = len(train_data)
    num_test_batches = len(test_data)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in enumerate(train_data, 0):
            optimizer.zero_grad(set_to_none=True)
            inputs, labels = inputs.to(device), labels.to(device).type(dtype=torch.int).view(-1, 1)
            if add_noise:
                inputs += torch.normal(inputs.max() / 50, inputs.max() / 50, inputs.size()).to(device)
            inputs -= inputs.min()
            inputs /= inputs.max()
            outputs = model(inputs).to(device).view(-1, num_classes, 1)
            loss = criterion(outputs, labels.long())
            pred = torch.argmax(outputs, 1)
            running_acc += torch.sum(pred == labels.data) / int(pred.size(0))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            printProgressBar(i + 1, num_batches, prefix = 'Epoch ' + str(epoch + 1) + '/' + str(epochs),
                             suffix = 'Batch {}/{} \t Mean Loss: {:.3f} \t Mean Acc: {:.3f}'.format(i + 1, num_batches,
                                                                                                    running_loss / (i + 1),
                                                                    running_acc / (i + 1)), length = 20)
        mean_loss = running_loss / num_batches
        mean_acc = running_acc / num_batches
        history['train_loss'].append(mean_loss)
        history['train_acc'].append(mean_acc)
        l = 0
        a = 0
        if epoch % val_iter:
            continue
        if test_data is None:
            continue
        v = [(j, 0) for j in d.values()]
        class_acc = dict(v)
        with torch.no_grad():
            
            class_counts = dict(v)
            class_right = dict(v)
            for i, (inputs, labels) in enumerate(test_data, 0):
                inputs, labels = inputs.to(device), labels.to(device).type(dtype=torch.int).view(-1, 1)
                inputs -= inputs.min()
                inputs /= inputs.max()
                outputs = model(inputs).to(device).view(-1, num_classes, 1)
                loss = criterion(outputs, labels.long())
                pred = torch.argmax(outputs, 1)
                acc = torch.sum(pred == labels.data) / int(pred.size(0))
                for j in range(len(pred)):
                    if pred[j, 0].item() == labels.data[j, 0].item():
                        class_right[d[pred[j, 0].item()]] += 1
                    class_counts[d[labels.data[j, 0].item()]] += 1
                l += float(loss.item())
                a += float(acc.item())
                printProgressBar(i + 1, num_test_batches, prefix = 'Epoch ' + str(epoch + 1) + '/' + str(epochs),
                             suffix = 'Batch {}/{} \t Mean Loss: {:.3f} \t Mean Acc: {:.3f}'.format(i + 1, num_test_batches,
                                                                                                    l / (i + 1),
                                                                    a / (i + 1)), length = 20)
        m_l = l / len(test_data)
        m_a = a / len(test_data)
        if m_a > best_acc:
            best_acc = m_a
            torch.save(model.state_dict(), path + '../' + model_name + '_weights.pth')
        history['val_loss'].append(m_l)
        history['val_acc'].append(m_a)
        #print("Summary: Epoch: {} \t Train Loss: {:.3f} \t Train Accuracy: {:.3f} \t Test Loss: {:.3f} \t Test Accuracy: {:.3f}"
              #.format(epoch + 1, mean_loss, mean_acc, m_l, m_a))
        for i in class_counts.keys():
            class_acc[i] = class_right[i] / class_counts[i]
        plt.figure(figsize=(10, 10))
        plt.bar(range(len(class_acc)), list(class_acc.values()), align='center')
        plt.xticks(range(len(class_acc)), list(class_acc.keys()), rotation=90)
        plt.show()

    return history


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, test_data):
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = []
        pred_proba = []
        figure = plt.figure(figsize=(30, 30))
        transform = torchvision.transforms.ToPILImage()
        tens = transforms.ToTensor()
        s = int(np.sqrt(len(test_data)))
        cols, rows = s, int(np.ceil(len(test_data) / s))
        new_img = []
        for img in test_data:
            img = img.to(device)
            img -= img.min()
            img /= img.max()
            outputs = model(img.view(1, 3, IMSIZE, IMSIZE)).to(device).view(-1, num_classes)
            proba = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy()
            pred_proba.append(proba)
            idx = torch.argmax(outputs).cpu().item()
            name = d[idx]
            out.append(name)
            img = transform(img)
            draw = ImageDraw.Draw(img)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 12)
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((10, 100),"{} with probability {:.2f}%".format(name, proba[0, idx] * 100),(255,255,255),font=font)
            new_img.append(tens(img))
            
        grid_img = torchvision.utils.make_grid(new_img)
        plt.axis("off")
        plt.imshow(grid_img.numpy().transpose(1, 2, 0), cmap="gray")
        plt.show()
    return out, pred_proba

def get_test_data(path):
    images = []
    transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((IMSIZE, IMSIZE))])
    for file in os.listdir(path):
        img = Image.open(path + file)
        img = transform(img)
        images.append(img)
    return images


if __name__ == "__main__":
    main()