# rotate-bb

## Dataset Preparation

The images and the labels for the training and validation set must be placed in separate directories. Corresponding to each image thereshould be a `txt` file.
The name of the text file should be the same as the name of the image. For example, the label corresponding to `0.png` should be `0.txt`.

The `.txt` file specifications are
- One row per object
- Each row in `x_min y_min x_max y_max d1 d2 h` format. 
