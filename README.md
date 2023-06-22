## vgg16-tiny-imagenet
CAU computer vision class project in Google Colab

## How to run
### 1. Make train_10class, test_10class folder
download tiny-imagenet-200 dataset and select 10 classes in wnids_10class.txt --> 
[dataset download](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
<br>
uncomment and execute the **dataset.py**
```bash
!python dataset.py
```

### 2. make log, checkpoint folder

### 3. run train and test code
```bash
!python train_test.py
```

### 4. tensorboard in terminal
```bash
 tensorboard --logdir='./log/' 
```
