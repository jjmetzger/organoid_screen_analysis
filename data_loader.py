from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import imageio
from glob import glob
from PIL import Image
import os


def dataloader_3chan_onedir(path, bs, shuffle, num_workers, normalize=True, return_filename=False):
    return DataLoader(
        dataset_onedir(path, transform=data_transform(normalize), return_filename=return_filename),
        batch_size=bs, shuffle=shuffle, num_workers=num_workers)


def data_transform(normalize=False):
    if normalize:
        dt = transforms.Compose([
                transforms.RandomRotation(180),
                transforms.ColorJitter(contrast=.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:
        dt = transforms.Compose([
                transforms.RandomRotation(180),
                transforms.ColorJitter(contrast=.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    return dt


class dataset_onedir(Dataset):

    def __init__(self, root_dir, transform=None, return_filename=False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_filename = return_filename

        self.filenames = glob(os.path.join(root_dir,'*.tif'))
        self.n_samples = len(self.filenames)
        if self.n_samples == 0:
            raise RuntimeError('No tif files found in directory ' + root_dir)
        else:
            print('found', self.n_samples, 'files')

        self.im_shape = imageio.imread(self.filenames[0]).shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        f = self.filenames[idx]
        image = imageio.imread(f)
        assert image.shape[0] == image.shape[1]
        if image.shape[0] % 2 == 1:  # if image size is odd, make even
            image = image[:-1,:-1]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if not self.return_filename:
            return image
        else:
            return image, f
