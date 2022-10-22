import time
import datetime
from collections import defaultdict

import torch
import torchvision
import cv2
from lenet import LeNet
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from math import log10, sqrt
import pandas as pd


class Benchmark:

    def __init__(self, pooling_function='max', dataset='cifar10', epochs=10):
        self.epochs = epochs
        self.num_classes = 10
        self.num_channels = 3
        self.batch_size = 100
        self.sample_image = None
        self.resultsdir = './results'
        self.pooling_function = pooling_function
        self.dataset = dataset

        self.dataloaders = defaultdict(DataLoader)
        self.load_data()

        self.model = LeNet(num_classes=self.num_classes, num_channels=self.num_channels,
                           pooling_function=pooling_function)

        self.device = 'cpu'
        if torch.cuda.is_available():
            print('Training on GPU')
            self.device = 'cuda'
        else:
            print('Training on CPU')

        self.model.to(self.device)

        self.loss_function = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def load_data(self):
        train_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        if self.dataset == 'cifar10':
            self.num_classes = 10
            train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True,
                                                     transform=train_transform)
            self.sample_image = train_set[0]
            self.dataloaders['train'] = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size,
                                                                    shuffle=True)
            test_set = torchvision.datasets.CIFAR10(root='./cifar', train=False, download=True,
                                                    transform=test_transform)
            self.dataloaders['val'] = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size,
                                                                  shuffle=False)
        if self.dataset == 'cifar100':
            self.num_classes = 100
            train_set = torchvision.datasets.CIFAR100(root='./cifar', train=True, download=True,
                                                      transform=train_transform)
            self.sample_image = train_set[0]
            self.dataloaders['train'] = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size,
                                                                    shuffle=True)
            test_set = torchvision.datasets.CIFAR100(root='./cifar', train=False, download=True,
                                                     transform=test_transform)
            self.dataloaders['val'] = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size,
                                                                  shuffle=False)
        if self.dataset == 'mnist':
            self.num_classes = 10
            self.num_channels = 1
            train_set = torchvision.datasets.MNIST(root='./mnist', train=True, download=True,
                                                   transform=train_transform)
            self.sample_image = train_set[0]
            self.dataloaders['train'] = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batch_size,
                                                                    shuffle=True)
            test_set = torchvision.datasets.MNIST(root='./mnist', train=False, download=True,
                                                  transform=test_transform)
            self.dataloaders['val'] = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size,
                                                                  shuffle=False)

    @staticmethod
    def format_time(elapsed):
        # Round to the nearest second.
        elapsed_rounded = int(round(elapsed))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def start_train(self):
        history = defaultdict(list)
        best_train_accuracy = 0
        best_val_accuracy = 0

        # save a sample image as original and pooled
        results_file = open(self.resultsdir + '/' + self.pooling_function + '-' + self.dataset + '_pooling.txt', 'w')
        self.imshow(torchvision.utils.make_grid(self.sample_image[0]), filename='original_image')
        output_images = self.model.all_layers_output(self.sample_image[0])
        for layer_name, image in output_images.items():
            if layer_name == 'initial':
                contrast = self.calc_contrast(torchvision.utils.make_grid(image))
                self.logwrite(f'Image RMS contrast is {contrast}', results_file=results_file)
                trans = transforms.Compose([transforms.Resize(32)])
                psnr = self.calc_psnr(torchvision.utils.make_grid(trans(image)),
                                      torchvision.utils.make_grid(self.sample_image[0]))
                self.logwrite(f'Image PSNR is {psnr}', results_file=results_file)
                ssim = self.calc_ssim(torchvision.utils.make_grid(trans(image)),
                                      torchvision.utils.make_grid(self.sample_image[0]))
                self.logwrite(f'Image SSIM is {ssim}', results_file=results_file)
            self.imshow(torchvision.utils.make_grid(image), filename='cnn_layer_' + layer_name)

        for epoch in range(self.epochs):
            t0 = time.time()
            self.logwrite(f'Epoch {epoch + 1}/{self.epochs} ', results_file=results_file)
            self.logwrite('-' * 10, results_file=results_file)
            train_loss, train_acc_1, train_acc_5 = self.train()
            self.logwrite(f'Train accuracy 1 {train_acc_1},5 {train_acc_5} ', results_file=results_file)
            if train_acc_1 > best_train_accuracy:
                best_train_accuracy = train_acc_1
            elapsed = self.format_time(time.time() - t0)
            self.logwrite('Train time elapsed: {:} '.format(elapsed), results_file=results_file)
            val_loss, val_acc_1, val_acc_5 = self.validate()
            self.logwrite(f'Validation accuracy 1 {val_acc_1}, 5 {val_acc_5} ', results_file=results_file)
            if val_acc_1 > best_val_accuracy:
                best_val_accuracy = val_acc_1
            elapsed = self.format_time(time.time() - t0)
            self.logwrite('Val time elapsed: {:}'.format(elapsed), results_file=results_file)
            history['time_elapsed'].append(elapsed)
            history['train_accuracy_1'].append(round(train_acc_1.item(), 4))
            history['train_accuracy_5'].append(round(train_acc_5.item(), 4))
            history['validation_accuracy_1'].append(round(val_acc_1.item(), 4))
            history['validation_accuracy_5'].append(round(val_acc_5.item(), 4))
        self.logwrite('Best val accuracy: {:}'.format(best_val_accuracy), results_file=results_file)
        self.logwrite('Best train accuracy: {:}'.format(best_train_accuracy), results_file=results_file)
        results_file.close()
        results = pd.DataFrame(history)
        results.index += 1
        results.to_csv(self.resultsdir + '/' + self.pooling_function + '_' + self.dataset + '_results_.csv',
                       index_label='Epoch')

    def logwrite(self, text, results_file):
        print(text)
        results_file.write(text)
        results_file.write('\n')

    def calc_contrast(self, img):
        if img.is_cuda:
            img = img.cpu().detach()
        npimg = img.numpy()
        img = np.transpose(npimg, (1, 2, 0))
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = img_grey.std()
        return contrast

    def calc_ssim(self, original, final):
        if original.is_cuda:
            original = original.cpu().detach()
        npimg = original.numpy()
        original = np.transpose(npimg, (1, 2, 0))
        if final.is_cuda:
            final = final.cpu().detach()
        npimg = final.numpy()
        final = np.transpose(npimg, (1, 2, 0))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        (ssim, diff) = structural_similarity(original, final, full=True)
        return ssim

    def calc_psnr(self, original, final):
        if original.is_cuda:
            original = original.cpu().detach()
        npimg = original.numpy()
        original = np.transpose(npimg, (1, 2, 0))
        if final.is_cuda:
            final = final.cpu().detach()
        npimg = final.numpy()
        final = np.transpose(npimg, (1, 2, 0))
        mse = np.mean((original - final) ** 2)
        if (mse == 0):
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def imshow(self, img, title=None, filename='image'):
        if img.is_cuda:
            img = img.cpu().detach()
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.savefig(self.resultsdir + '/' + self.pooling_function + '_' + self.dataset + '_' + filename + '.png',
                    bbox_inches="tight")
        # plt.show()

    def train(self):
        self.model.train()
        train_loss = 0
        correct_1 = 0
        correct_5 = 0

        for batch_num, (data, target) in enumerate(self.dataloaders['train']):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predictions = output.topk(5, 1, True, True)
            predictions = predictions.t()
            correct = predictions.eq(target.view(1, -1).expand_as(predictions))

            correct_1 += correct[:1].reshape(-1).float().sum(0)
            correct_5 += correct[:5].reshape(-1).float().sum(0)

        return train_loss, correct_1 / len(self.dataloaders['train'].dataset), correct_5 / len(
            self.dataloaders['train'].dataset)

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct_1 = 0
        correct_5 = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.dataloaders['val']):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_function(output, target)
                test_loss += loss.item()
                _, predictions = output.topk(5, 1, True, True)
                predictions = predictions.t()
                correct = predictions.eq(target.view(1, -1).expand_as(predictions))

                correct_1 += correct[:1].reshape(-1).float().sum(0)
                correct_5 += correct[:5].reshape(-1).float().sum(0)

        return test_loss, correct_1 / len(self.dataloaders['val'].dataset), correct_5 / len(
            self.dataloaders['val'].dataset)
