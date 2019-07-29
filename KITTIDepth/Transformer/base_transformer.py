import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseTransformer(ABC):
    """
    This class is an abstract transformer, one can override transform methods.
    The subclass need the implementation of the following methods.
    """
    def __init__(self, phase):
        self.phase = phase

    @abstractmethod
    def get_joint_transform(self):
        pass

    @abstractmethod
    def get_img_transform(self):
        pass

    @abstractmethod
    def get_depth_transform(self):
        pass

    def get_transform(self):
        """
        Total transform processing
        """
        joint_transform = self.get_joint_transform()
        img_transform = self.get_img_transform()
        depth_transform= self.get_depth_transform()

        return transforms.Compose([joint_transform,
                                   img_transform,
                                   depth_transform])
