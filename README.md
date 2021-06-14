# rotate-bb

## Dataset Preparation

The images and the labels for the training and validation set must be placed in separate directories. Corresponding to each image thereshould be a `txt` file.
The name of the text file should be the same as the name of the image. For example, the label corresponding to `0.png` should be `0.txt`.

The `.txt` file specifications are
- One row per object
- Each row in `x_min y_min x_max y_max d1 d2 h` format. `x_min y_min x_max y_max` are the coordinates of the horizontal bounding box in the image. `d1` and `d2` are the offsets from the top left corner, where the rotated bounding box meets the horizontal bounding box. `h` is the length of the rotated bounding box away from the offsets. Refer to (https://arxiv.org/pdf/2101.07383.pdf) for the definitions of `d1`,`d2` and `h`.

## Training the network

To train the network, use
python train.py --train_image_paths [path to the train images] --train_label_paths [path to the train labels] --test_image_paths [path to the validation images] --test_label_paths [path to the validation labels].

