from __future__ import division
import os

import torch
import torchvision

#import cub200
import pdb
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable
from CompactBilinearPooling1 import CompactBilinearPooling

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 512
input_dim1 = 512
input_dim2 = 512
output_dim = 16384

generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]), rand_s.float(), [input_dim, output_dim]).to_dense()
sketch_matrix01 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim1,)), 2 * torch.randint(2, size = (input_dim1,)) - 1, input_dim1, output_dim))
sketch_matrix02 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim2,)), 2 * torch.randint(2, size = (input_dim2,)) - 1, input_dim2, output_dim))
torch.save(sketch_matrix01,'/cvdata/xuelu/CUB_200_2011/bilinear-cnn1/src/sketch_matrix1_16384.pth')
torch.save(sketch_matrix02,'/cvdata/xuelu/CUB_200_2011/bilinear-cnn1/src/sketch_matrix2_16384.pth')

class BCNN(torch.nn.Module):
    """B-CNN for CUB200.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(output_dim, 200)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of shape N*3*448*448.

        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """

#        pdb.set_trace()
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        sketch_matrix1 = sketch_matrix01.cuda()
        sketch_matrix2 = sketch_matrix02.cuda()
        fft1 = torch.rfft(X.permute(0, 2, 3, 1).matmul(sketch_matrix1), 1)
        fft2 = torch.rfft(X.permute(0, 2, 3, 1).matmul(sketch_matrix2), 1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        cbp = torch.irfft(fft_product, 1, signal_sizes = (output_dim,)) * output_dim
#        pdb.set_trace()
#        XX = X.permute(0, 2, 3, 1).contiguous().view(-1, 512)
        
#        X = torch.matmul(X,X.transpose(1,2))/(28**2)  # Bilinear
#        assert X.size() == (N, 512, 512)
#        X = X.view(N, 512**2)
        X = cbp.sum(dim = 1).sum(dim = 1)
        X = torch.sqrt(F.relu(X)) - torch.sqrt(F.relu(-X))
        X = torch.nn.functional.normalize(X)
#        pdb.set_trace()
        X = self.fc(X)        
        assert X.size() == (N, 200)
        return X


class BCNNManager(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _options: Hyperparameters.
        _path: Useful paths.
        _net: Bilinear CNN.
        _criterion: Cross-entropy loss.
        _solver: SGD with momentum.
        _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
        _train_loader: Training data.
        _test_loader: Testing data.
    """
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
#        pdb.set_trace()
        # Network.
        self._net = torch.nn.DataParallel(BCNN()).cuda()
        # Load the model from disk.
#        self._net.load_state_dict(self.load_my_state_dict(torch.load(self._path['model'])))
#        self._net.load_state_dict(torch.load(self._path['model']))
        print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        self._solver = torch.optim.SGD(
            self._net.module.fc.parameters(), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=5, verbose=True,
            threshold=1e-4)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        data_dir1 = '/cvdata/xuelu/CUB_200_2011/train'
        train_data = datasets.ImageFolder(root=data_dir1,transform=train_transforms)
        data_dir2 = '/cvdata/xuelu/CUB_200_2011/test'
        test_data = datasets.ImageFolder(root=data_dir2,transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=8,
            shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            u = 0
            for X, y in self._train_loader:
                u = u+1
#                if u == 375:
#                   pdb.set_trace()
                # Data.
                X = torch.autograd.Variable(X.cuda())
                y = torch.autograd.Variable(y.cuda(async=True))

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
#                pdb.set_trace()
                loss = self._criterion(score, y)
                print('| Epoch %2d Iter %3d\tBatch loss %.4f\t' % (t+1,u,loss))
                epoch_loss.append(loss.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
                # Backward pass.
                loss.backward()
                self._solver.step()

            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
#                print('*', end='')
# Save model onto disk.
                torch.save(self._net.state_dict(),
                           os.path.join('/cvdata/xuelu/CUB_200_2011/bilinear-cnn1/src/model',
                                        'd16384_vgg_16_epoch_%d.pth' % (t + 1)))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.train(False)
        torch.no_grad()
        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            # Data.

            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda(async=True))
#            X = X.to(device)
#            y = y.to(device)

            # Prediction.
#            pdb.set_trace()
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data).item()
#            del X, y
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total

def main():
    """The main function."""

    options = {
        'base_lr': 1,
        'batch_size': 16,
        'epochs': 100,
        'weight_decay': 1e-5,
    }

    project_root = os.popen('pwd').read().strip()
    path = {
#        'cub200': os.path.join(project_root, 'data/cub200'),
        'model': os.path.join(project_root, 'model'),
    }
    for d in path:
        assert os.path.isdir(path[d])
#    pdb.set_trace()
#        else:
#            assert os.path.isdir(path[d])

    manager = BCNNManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()
