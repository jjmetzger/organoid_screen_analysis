import torch
from torch import nn
from torch.autograd import Variable
import data_loader as dl
import os
import numpy as np
import time
import argparse
import pathlib


class autoencoder(nn.Module):
    def __init__(self, image_shape, latent_dim=64, act_type='elu', kernel_size=3, pad=1, dilation=1, stride=1, kernel_size_mp=2):
        super(autoencoder, self).__init__()
        if act_type == 'elu':
            act_layer = nn.ELU(inplace=True)
        elif act_type == 'relu':
            act_layer = nn.ReLU(inplace=True)
        else:
            raise RuntimeError('unknown activation layer')

        self.linear_layer_shape, intermediate_shape_even = self.determine_conv_shapes(image_shape, pad, dilation, kernel_size, stride, kernel_size_mp)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size, stride=stride, padding=pad),
            act_layer,
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size, stride=stride, padding=pad),
            act_layer,
            nn.MaxPool2d(2),
        )

        self.latent = nn.Sequential(
            nn.Linear(self.linear_layer_shape * self.linear_layer_shape * 64, latent_dim),
            act_layer
        )

        self.x_decoded_dense1 = nn.Linear(latent_dim, self.linear_layer_shape*self.linear_layer_shape*64)
        self.x_decoded_dens1_act = act_layer
    
        self.latent = nn.Sequential(
            nn.Linear(self.linear_layer_shape*self.linear_layer_shape*64, latent_dim),
            act_layer,
        )

        self.x_decoded_dense1 = nn.Linear(latent_dim, self.linear_layer_shape*self.linear_layer_shape*64)
        self.x_decoded_dens1_act = act_layer

        self.x_decoded_upsample0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size, stride=stride+1, padding=intermediate_shape_even, output_padding=intermediate_shape_even),
            act_layer,
            nn.BatchNorm2d(32),
        )

        self.x_decoded_upsample3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size, stride=stride+1, padding=pad, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.linear_layer_shape*self.linear_layer_shape*64)
        x_latent = self.latent(x)
        x = self.x_decoded_dense1(x_latent)
        x = x.view(-1, 64, self.linear_layer_shape, self.linear_layer_shape)
        x = self.x_decoded_dens1_act(x)
        x = self.x_decoded_upsample0(x)
        x = self.x_decoded_upsample3(x)

        return x, x_latent

    @staticmethod
    def determine_conv_shapes(image_shape, pad, dilation, kernel_size, stride, kernel_size_mp):
        # size of linear layer depends on the image input size
        shape = image_shape[0]
        shape = int(np.floor((shape + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1))
        shape = int(np.floor((shape - (kernel_size_mp-1) -1)/kernel_size_mp + 1))
        intermediate_shape_even = 1-(shape % 2)
        shape = int(np.floor((shape + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1))
        shape = int(np.floor((shape - (kernel_size_mp-1) -1)/kernel_size_mp + 1))

        return shape, intermediate_shape_even


def run(datadir, num_epochs, batch_size, lr, latent_dim, weight_decay, activation_type, output_folder, kernel_size):

    image_data = dl.dataloader_3chan_onedir(datadir, bs=batch_size, shuffle=True, num_workers=8, normalize=False)
    image_shape = image_data.dataset[0].shape[1:]
    model = autoencoder(image_shape=image_shape, latent_dim=latent_dim, act_type=activation_type, kernel_size=kernel_size)

    if torch.cuda.is_available():
        model = model.cuda()

    loss_over_time = []

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for data in image_data:
            img = data
            if torch.cuda.is_available():
                img = Variable(img).cuda()
            else:
                img = Variable(img)

            output, latent_vec = model(img)
            loss = loss_function(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch {epoch+1}/{num_epochs}, loss {loss.data:.4f}')
        loss_over_time.append(loss.data)

    torch.save(model.state_dict(), os.path.join(output_folder, 'ae_weights.pth'))
    return loss_over_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="directory in which the images are located")
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=200)
    parser.add_argument("--batch_size", type=int, help="batch size", default=4)
    parser.add_argument("--lr", type=float, help="learning rate", default=2e-5)
    parser.add_argument("--latent_dim", type=int, help="dimension of the latent space", default=64)
    parser.add_argument("--weight_decay", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--activation_type", type=str, help="activation type", default='elu')
    parser.add_argument("--kernel_size", type=int, help="convolutional layer kernel size", default=3)
    parser.add_argument("--output_folder", type=str, help="output folder")
    parser.add_argument("-loss_vs_time_file", type=str, help="file in which to save the loss", default="loss_vs_time.txt")

    args = parser.parse_args()

    if args.output_folder is None:
        args.output_folder = str(int(round(time.time())))  # current time stamp

    pathlib.Path(args.output_folder).mkdir(exist_ok=True)
    print('writing results to', args.output_folder)

    loss_over_time_full = run(args.datadir, args.num_epochs, args.batch_size, args.lr, args.latent_dim, args.weight_decay,
                              args.activation_type, args.output_folder, args.kernel_size)

    np.savetxt(os.path.join(args.output_folder, args.loss_vs_time_file),
               np.array(loss_over_time_full))




