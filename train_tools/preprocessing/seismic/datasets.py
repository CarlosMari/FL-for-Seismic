
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class InlineLoader(Dataset):
    """Dataset class for loading F3 patches and labels"""

    def __init__(self, seismic_cube, label_cube, inline_inds, train_status=True, transform=None, preprocessing=None):
        """Initializer function for the dataset class

        Parameters
        ----------
        seismic_cube: ndarray, (inline, crossline, depth)
                    3D ndarray of floats representing seismic amplitudes

        label_cube: ndarray, shape (inline, crossline, depth)
                 3D ndarray same dimensions as seismic_cube containing label information.
                 Each value is [0,num_classes]

        inline_inds: ndarray of ints
                    A list of Integers specifying the indices of the inline sections to train on

        train_status: bool,
                    bool to specify if loader being used in training mode or not.

        """

        self.seismic = seismic_cube
        self.label = label_cube
        self.indices = inline_inds
        self.train_status = train_status
        self.transform = transform
        self.label_transform = transforms.Compose([transforms.ToTensor()])
        self.preprocessing = preprocessing
        self.num_sections = len(inline_inds)

    def __getitem__(self, index):
        """Obtains the image crops relating to each section in the given orientation.

        Parameters
        ----------
        index: int
             Integer specifies the section number along the given orientation.

        Returns
        -------
        images: ndarray of shape (1, H, W)
              Returns inline section specified by index"""

        inline_num = self.indices[index]

        section = self.seismic[:, inline_num, :].T # gets all inline for particular crossline

        label_section = self.label[:, inline_num, :].T

        if self.preprocessing:
            s = self.preprocessing(image=section, mask=label_section)
            section, label_section = s['image'], s['mask']
        # apply transforms
        if self.transform:
            section = self.transform(section)

        if self.train_status == False:
            return section, label_section

        return section, label_section

    def __len__(self):
        """Retrieves total number of training samples"""

        return self.num_sections