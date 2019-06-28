# KITTITools
Tools for [KITTI depth prediction](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) data processing.

This repo provides two useful modules for loading and preprocessing KITTI depth data set.

### Dataloader

The module **Dataloader** provides an interface to load data items from eigen split.

Here is an example to use it:

```python
from Dataloader import Kittiloader
...

loader = Kittiloader(<kitti_root_path>, <'train', 'val' or 'test'>)
data_size = loader.data_length() # get data split size
data_item = loader.load_item(idx) # which is very suitable for pytorch dataloader
```

These methods helps user to define a pytorch Dataset in a more convenient way.

### Transformer

The custom_xxx.py files can be modified by users to define their own transformers in practice.

Just follow the base model, one can construct a variety of of pytorch DataLoaders quickly.

An example is included in this module, which works well with dataset.py, which executes standard and the most straightforward pytorch DataLoader generation steps.

To use the given data loader, try the following code:

```python
from dataset import DataGenerator
...

# transformer will be defined automatically according to phase once datagen instance is created
datagen = DataGenerator(<kitti_root_path>, phase=<'train', 'val' or 'test'>)
kittidataset = datagen.create_data(batch_size)

# other code before training loop
...

for epoch in range(num_epoches):
# training loop for an epoch
    for id, batch in enumerate(kittidataset):
        # here one an get various types of data
        left_img_batch = batch['left_img'] # batch of left image, id 02
        right_img_batch = batch['right_img'] # batch of right image, id 03
        depth_batch = batch['depth'] # the corresponding depth ground truth of given id
        depth_interp_batch = batch['depth_interp'] # dense depth for visualization
        fb = batch['fb'] # focal_length * baseline

        # training code
        ...
```
