from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


class BaseMethod(object):
    """
    The base class to generate various data transform subclass.
    One can define his own transform methods as custom methods.
    """
    def __init__(self, mode=""):
        self.mode=mode

    def set_data(self, data_item):
        self.left_img = data_item['left_img']
        self.right_img = data_item['right_img']
        self.depth = data_item['depth']
        self.depth_interp = data_item['depth_interp']
        self.fb = data_item['fb']

    @staticmethod
    def _is_pil_image(img):
        if accimage is not None:
            return isinstance(img, (Image.Image, accimage.Image))
        else:
            return isinstance(img, Image.Image)
