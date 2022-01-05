```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    

## Autoencoder (encoder-decoder) for X-ray Image
Here we will design an autoencoder to reconstruct 3D image from 2D X-ray images. First, we will feed the X-ray images to the network. In 3D reconstruction, our target will be a volume of images. 

So, <br>input of the model: 2D X-ray images 
<br>target of the model: Volume of the images

This code is used to generate 3D vertebrae from 2D X-Ray images of vertebrae.

The network can be divided into three sections -

                                   Representation network 
                                   Transformation network
                                   Generation network

Representation network: It downsamples the feature maps. But, while downsampling, we will increase the no. of feature maps by increasing the no. of convolutional filters.

Transformation network: It converts the 2D feature maps into 3D feature maps.

Generation network: Here we will upsample the feature maps and finally will convert to a stack of 2D images.

v5 Update:
- Checkpoint is introduced
- Reducing learning rate is introduced

v6 Update:
- Plotting introduced

v7 Update:
- training dataset is resized along the z-axis

v7_5 Update:
- Moving to new format: channel x depth x height x width. (Previously it was - depth x height x width x channel)
- Channel first format

v10_1 Update:
- This code is for segmented vertebrae. New code for data preprocessing is introduced

v10_2 Update:
- Image stacking is introduced

v10_3 Update:
- Image stacking and not-stacking are integrated together

v10_4 Update:
- New boolean variable `stack` is introduced. For multiple projections, `stack` is `True`. 

v10_5 Update:
- Custom loss function is introduced. It considers both `mse` and `psnr` in calculating loss.

v10_6 Update:
- Custom loss functions are added in `monitor`

v10_7 Update:
- two axes projections are used during training

v10_8 Update:
- Created a user friendly function that can handle both single and multi projections
- shuffling of image names added


```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Conv3D, UpSampling2D, MaxPooling2D, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import BatchNormalization, Add, Reshape
from tensorflow.keras.backend import squeeze, transpose, reshape
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import cv2
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio
import scipy.io as sio

%matplotlib inline
```

### Prepare training images and corresponding targets

Data and targets will be preprocessed using the funtion `create_data`. Details of it will be found in `prepareData.py`. 


```python
# Parameters 
# filters = [256, 512, 1024, 2048, 4096, 128, 64] # 128 and 64 are used only for deconv
filters = [192, 384, 768, 1536, 3072, 96, 48]
# filters = [128, 256, 512, 1024, 2048, 64, 32]
# filters = [32, 64, 128, 256, 512, 16, 8]

# depth = 120            # Depth means images along the z-axis
```

Load the training images saved in `.mat` format. It is created using `readImage.py`

Set directory


```python
'''
Image location information:
X-ray images are stored in the following way -

    xray -------> train --> several vertebrae folders --> images inside each folder
       |
        --------> test --> several vertebrae folders --> images inside each folder
    
CTs are stored in the following way - 
    ct -------> train --> several .mat files
     |
      --------> test --> several .mat files
'''
# xRay location
trainDir = '/content/drive/My Drive/3Dreconstruction/xray/vertebra/xray-x-y-axes' #vertebra-xray

# Target (CT) location
trainCtDir = '/content/drive/My Drive/3Dreconstruction/ct/vertebra/train'  #vertebra-CT

#************ Uncomment if you want to create trainSubset AUTOMATICALLY
# trainSubset = [item for item in os.listdir(trainDir)]   # folders inside the train folder
# trainSubset = trainSubset[0:30]
#************ Uncomment if you want to create trainSubset MANUALLY
trainSubset = ['vertebra08','vertebra09','vertebra11','vertebra12','vertebra13','vertebra14']

print(trainSubset)
print(len(trainSubset))
```

    ['vertebra08', 'vertebra09', 'vertebra11', 'vertebra12', 'vertebra13', 'vertebra14']
    6
    

Create projection indices which will be stacked together later



```python
# Create projection indices which will be stacked together later
'''
Let's say, we want to create a series of stacks as follows - 
[
['x_axis_deg_000', 'x_axis_deg_030', 'x_axis_deg_060'], ['x_axis_deg_010', 'x_axis_deg_040', 'x_axis_deg_070'], ......
['y_axis_deg_000', 'y_axis_deg_030', 'x_axis_deg_060'], ['y_axis_deg_010', 'y_axis_deg_040', 'y_axis_deg_070'], ......
]

So, first we need the axes names that we will be using. We define them in a list called 'axis_name'.

Then, we need a starting projection angle. We call it 'startAngle'. For the above example, startAngle = 0

We then need to know how many projections are we going to stack together. We call it 'noOfPrjtn'. For the above example,
it is 3. So, each stack will have 3 projections inside. 

Next, we need to increment the projection angle for the existing stack. We call it 'prjtnAngleIncr'. For the above example, 
it is 30. So, the projections are 0-deg, 30-deg, and 60-deg. 

Now, what will be the starting projection for the next stack? We define it as 'previous starting angle + an increment'.
The 'incr' variable is used to do so. For the above example, it is 10. So, the starting angle for the 2nd stack will be
0 + 10 = 10. Likewise, for the 3rd stack, it will be 10+10 = 20

Also, we need to define a stopping critera to stop making stacks. We did this using 'maxStartAngle'. For instance, when 
maxStartAngle is set to 90, it means that while creating a stack whose starting angle is larger than 90, it will
not create that stack, rather will exit from stacking.  

We have also added an error message if any angle becomes more than 360-deg.

Finally, x-ray images are stored in the following format: x_axis_deg_xxx.png. For instance: x_axis_degree_125.png'. 
So, we need to create this 3-digit projection angle. We did this using 'zfill'.
'''
# axis_name = ['x_axis_deg_', 'y_axis_deg_']
axis_name = ['x_axis_deg_']
startAngle = 0                                     
noOfPrjtn = 1
prjtnAngleIncr = 90           # Increase projection angle within the stack
incr = 10                     # Increase starting angle for the next stack                                           
maxStartAngle = 350            # Max starting angle allowed for stacking

stackIdx = []
temp = []

storeIntStartAngle = startAngle                                     # Store initial starting angle
for axis in axis_name:
  while (startAngle <= maxStartAngle):
    moveAngle = startAngle
    for i in range(noOfPrjtn):
      strName = axis + str(moveAngle).zfill(3)                      # Convert it to 3-digit
      temp.append(strName)
      assert moveAngle <= 360, "Angle should not cross 360 degree"  # Error msg, if any angle>360
      moveAngle += prjtnAngleIncr

    startAngle += incr

    stackIdx.append(temp)
    temp = []
  startAngle = storeIntStartAngle

print('No. of training images per object: ', len(stackIdx))
print('No. of objects: ', len(trainSubset))
print('Total no. of training images: ', len(stackIdx)*len(trainSubset))
print(stackIdx)

```

    No. of training images per object:  36
    No. of objects:  6
    Total no. of training images:  216
    [['x_axis_deg_000'], ['x_axis_deg_010'], ['x_axis_deg_020'], ['x_axis_deg_030'], ['x_axis_deg_040'], ['x_axis_deg_050'], ['x_axis_deg_060'], ['x_axis_deg_070'], ['x_axis_deg_080'], ['x_axis_deg_090'], ['x_axis_deg_100'], ['x_axis_deg_110'], ['x_axis_deg_120'], ['x_axis_deg_130'], ['x_axis_deg_140'], ['x_axis_deg_150'], ['x_axis_deg_160'], ['x_axis_deg_170'], ['x_axis_deg_180'], ['x_axis_deg_190'], ['x_axis_deg_200'], ['x_axis_deg_210'], ['x_axis_deg_220'], ['x_axis_deg_230'], ['x_axis_deg_240'], ['x_axis_deg_250'], ['x_axis_deg_260'], ['x_axis_deg_270'], ['x_axis_deg_280'], ['x_axis_deg_290'], ['x_axis_deg_300'], ['x_axis_deg_310'], ['x_axis_deg_320'], ['x_axis_deg_330'], ['x_axis_deg_340'], ['x_axis_deg_350']]
    

Shuffle stackIdx (Optional)</br>
- If you want to shuffle stackIdx, set the `shuffle` variable to `True`


```python
shuffle = True
if shuffle is True:
  perm = np.random.permutation(len(stackIdx))
  shuffleStackIdx = []
  for i in range(len(stackIdx)):
    shuffleStackIdx.append(stackIdx[perm[i]])
  stackIdx = shuffleStackIdx

print('No. of training images per vertebra: ', len(stackIdx))
print(stackIdx)
```

    No. of training images per vertebra:  36
    [['x_axis_deg_220'], ['x_axis_deg_060'], ['x_axis_deg_250'], ['x_axis_deg_150'], ['x_axis_deg_210'], ['x_axis_deg_350'], ['x_axis_deg_330'], ['x_axis_deg_300'], ['x_axis_deg_260'], ['x_axis_deg_070'], ['x_axis_deg_340'], ['x_axis_deg_100'], ['x_axis_deg_240'], ['x_axis_deg_160'], ['x_axis_deg_000'], ['x_axis_deg_110'], ['x_axis_deg_040'], ['x_axis_deg_080'], ['x_axis_deg_190'], ['x_axis_deg_180'], ['x_axis_deg_010'], ['x_axis_deg_140'], ['x_axis_deg_290'], ['x_axis_deg_130'], ['x_axis_deg_050'], ['x_axis_deg_200'], ['x_axis_deg_310'], ['x_axis_deg_230'], ['x_axis_deg_120'], ['x_axis_deg_020'], ['x_axis_deg_170'], ['x_axis_deg_280'], ['x_axis_deg_270'], ['x_axis_deg_090'], ['x_axis_deg_030'], ['x_axis_deg_320']]
    

Create a function to append xrays and CTs


```python
# Create a function to append xrays and CTs
'''
The following function is to store xray images and their corresponding targets.
The output is a list --> [xray, ct]

imgSize is an optional argument. If you want to resize then use imgSize.
For example, imgSize = (128,128)
'''
def create_data(trainDir, trainCtDir, trainSubset, stkIdx=None, imgSize=None):
    output = []
    noOfPrjtn = len(stkIdx[0])            # No. of projections in a single stack

    for subset in trainSubset:
        matName = subset + '.mat'
        Ct = sio.loadmat(os.path.join(trainCtDir, matName))
        key = sorted(Ct.keys())
        trainCt = Ct[key[3]]  

        for idx in range(len(stkIdx)):    # Looping around no. of stacks
          stkImg = []                     # This will append all projections for a single stack 
          for i in range(noOfPrjtn):      # Looping around no. projections in a stack
            ext = '.png'                  # Extension
            stkName = stkIdx[idx][i] + ext 
            if (ext == '.png'):          
              stk = cv2.imread(os.path.join(trainDir, subset, stkName), 0)
            elif (ext == '.mat'):
              loadStk = sio.loadmat(os.path.join(trainDir, subset, stkName))
              key = sorted(loadStk.keys())
              stk = loadStk[key[3]]

            if imgSize is not None:
              stk = cv2.resize(stk, imgSize)
            stkImg.append(stk)
          
          # stkImg = np.array(stkImg, dtype='uint8')
          stkImg = np.array(stkImg)
          output.append([stkImg, trainCt])           

    return output            
```

Prepare training data


```python
# Prepare training data
# imgSize = (128,128)    
train = create_data(trainDir, trainCtDir, trainSubset, stackIdx, imgSize=None)  # For stacked multiple projections

X_train = []
Y_train = []

for feature, gt in train:
    X_train.append(feature)
    Y_train.append(gt)
    
X_train = np.array(X_train)                 

if len(X_train.shape) < 4:
  X_train = X_train[:,np.newaxis,:,:]       # Creating channel first

Y_train = np.array(Y_train)
Y_train = np.moveaxis(Y_train, -1, 1)       # Creating channel first

print('Shape of X_train: ', X_train.shape)  # Shape: N, viewIdx or channel, sizeX, sizeY
print('Shape of Y_train: ', Y_train.shape)

depth = Y_train.shape[1]
print('Depth of the CT: ', depth)
```

    Shape of X_train:  (216, 1, 128, 128)
    Shape of Y_train:  (216, 50, 128, 128)
    Depth of the CT:  50
    


```python
# Normalize the feature
# X_train = X_train/255 
X_train = X_train - np.min(X_train)
X_train = X_train / np.max(X_train)
```


```python
# Normalize the target
# target = Y_train/255
target = Y_train
target = target - np.min(target)
target = target / np.max(target)
```

## Encoder-Decoder (Autoencoder)
- We will use functional API

## Representation network


```python
inputImg = Input(shape=X_train.shape[1:], name='input_img') #define input layer

##############################################################################
# 1st conv layer
conv1 = Conv2D(filters[0], (4,4), strides=(2,2), padding='same', name='conv1', data_format='channels_first')(inputImg) 
conv1_BN = BatchNormalization()(conv1) 
conv1_out = Activation('relu')(conv1_BN)

# 2nd conv layer
conv2 = Conv2D(filters[0], (3,3), strides=(1,1), padding='same', name='conv2', data_format='channels_first')(conv1_out)
conv2_BN = BatchNormalization()(conv2)

# Add conv1 and conv2_BN (shortcut path)
add_conv1_2 = Add()([conv1_out, conv2_BN])

# Residual output of 1st and 2nd layers
conv2_out = Activation('relu')(add_conv1_2)

###############################################################################
# 3rd conv layer
conv3 = Conv2D(filters[1], (4,4), strides=(2,2), padding='same', name='conv3', data_format='channels_first')(conv2_out) 
conv3_BN = BatchNormalization()(conv3) 
conv3_out = Activation('relu')(conv3_BN)

# 4th conv layer
conv4 = Conv2D(filters[1], (3,3), strides=(1,1), padding='same', name='conv4', data_format='channels_first')(conv3_out)
conv4_BN = BatchNormalization()(conv4)

# Add conv3 and conv4_BN (shortcut path)
add_conv3_4 = Add()([conv3_out, conv4_BN])

# Residual output of 3rd and 4th layers
conv4_out = Activation('relu')(add_conv3_4)

###############################################################################
# 5th conv layer
conv5 = Conv2D(filters[2], (4,4), strides=(2,2), padding='same', name='conv5', data_format='channels_first')(conv4_out) 
conv5_BN = BatchNormalization()(conv5) 
conv5_out = Activation('relu')(conv5_BN)

# 6th conv layer
conv6 = Conv2D(filters[2], (3,3), strides=(1,1), padding='same', name='conv6', data_format='channels_first')(conv5_out)
conv6_BN = BatchNormalization()(conv6)

# Add conv5 and conv6_BN (shortcut path)
add_conv5_6 = Add()([conv5_out, conv6_BN])

# Residual output of 5th and 6th layers
conv6_out = Activation('relu')(add_conv5_6)

###############################################################################
# 7th conv layer
conv7 = Conv2D(filters[3], (4,4), strides=(2,2), padding='same', name='conv7', data_format='channels_first')(conv6_out) 
conv7_BN = BatchNormalization()(conv7) 
conv7_out = Activation('relu')(conv7_BN)

# 8th conv layer
conv8 = Conv2D(filters[3], (3,3), strides=(1,1), padding='same', name='conv8', data_format='channels_first')(conv7_out)
conv8_BN = BatchNormalization()(conv8)

# Add conv7 and conv8_BN (shortcut path)
add_conv7_8 = Add()([conv7_out, conv8_BN])

# Residual output of 7th and 8th layers
conv8_out = Activation('relu')(add_conv7_8)

###############################################################################
# 9th conv layer
conv9 = Conv2D(filters[4], (4,4), strides=(2,2), padding='same', name='conv9', data_format='channels_first')(conv8_out) 
conv9_BN = BatchNormalization()(conv9) 
conv9_out = Activation('relu')(conv9_BN)

# 10th conv layer
conv10 = Conv2D(filters[4], (3,3), strides=(1,1), padding='same', name='conv10', data_format='channels_first')(conv9_out)
conv10_BN = BatchNormalization()(conv10)

# Add conv9 and conv10_BN (shortcut path)
add_conv9_10 = Add()([conv9_out, conv10_BN])

# Residual output of 9th and 10th layers
conv10_out = Activation('relu')(add_conv9_10)
```


```python
# Create autoencoder representation model
autoEncRep = Model(inputs=inputImg, outputs=conv10_out, name='xray_autoencoder_representation_model')
```


```python
# autoEncRep.summary()
```

## Transformation network


```python
trans1 = Conv2D(filters[4], (1,1), padding='same', activation='relu', name='trans1', data_format='channels_first')(conv10_out)
trans1_reshape = Reshape([filters[3],2,trans1.shape[2],trans1.shape[3]])(trans1)
trans_out = Conv3D(filters[3], kernel_size=1, strides=(1,1,1), padding='same', activation='relu', name='trans_out', data_format='channels_first')(trans1_reshape)

```


```python
autoEncTrans = Model(inputs=inputImg, outputs=trans_out, name='xray_autoencoder_transformation_model')
```


```python
# autoEncTrans.summary()
```


```python
trans_out
```




    <tf.Tensor 'trans_out/Relu:0' shape=(None, 1536, 2, 4, 4) dtype=float32>



## Generation network


```python
##########Deconvolution###############################
#1024
gen1 = Conv3DTranspose(filters[2], kernel_size=4, strides=(2,2,2), padding='same', name='gen1', data_format='channels_first')(trans_out)
gen1_BN = BatchNormalization()(gen1) 
gen1_out = Activation('relu')(gen1_BN)

#512
gen2 = Conv3DTranspose(filters[1], kernel_size=4, strides=(2,2,2), padding='same', name='gen2', data_format='channels_first')(gen1_out)
gen2_BN = BatchNormalization()(gen2) 
gen2_out = Activation('relu')(gen2_BN)

gen3 = Conv3DTranspose(filters[1], kernel_size=3, strides=(1,1,1), padding='same', name='gen3', data_format='channels_first')(gen2_out)
gen3_BN = BatchNormalization()(gen3) 
gen3_out = Activation('relu')(gen3_BN)

#256
gen4 = Conv3DTranspose(filters[0], kernel_size=4, strides=(2,2,2), padding='same', name='gen4', data_format='channels_first')(gen3_out)
gen4_BN = BatchNormalization()(gen4) 
gen4_out = Activation('relu')(gen4_BN)

gen5 = Conv3DTranspose(filters[0], kernel_size=3, strides=(1,1,1), padding='same', data_format='channels_first')(gen4_out)
gen5_BN = BatchNormalization()(gen5) 
gen5_out = Activation('relu')(gen5_BN)

#128
gen6 = Conv3DTranspose(filters[5], kernel_size=4, strides=(2,2,2), padding='same', name='gen6', data_format='channels_first')(gen5_out)
gen6_BN = BatchNormalization()(gen6) 
gen6_out = Activation('relu')(gen6_BN)

gen7= Conv3DTranspose(filters[5], kernel_size=3, strides=(1,1,1), padding='same', data_format='channels_first')(gen6_out)   #128
gen7_BN = BatchNormalization()(gen7) 
gen7_out = Activation('relu')(gen7_BN)

#64
gen8 = Conv3DTranspose(filters[6], kernel_size=4, strides=(2,2,2), padding='same', name='gen8', data_format='channels_first')(gen7_out)
gen8_BN = BatchNormalization()(gen8) 
gen8_out = Activation('relu')(gen8_BN)

gen9= Conv3DTranspose(filters[6], kernel_size=3, strides=(1,1,1), padding='same', data_format='channels_first')(gen8_out)    #64
gen9_BN = BatchNormalization()(gen9) 
gen9_out = Activation('relu')(gen9_BN)

#############Transforming#####################
gen10 = Conv3DTranspose(1, kernel_size=1, padding='same', name='g10', data_format='channels_first')(gen9_out)
gen11 = squeeze(gen10, 1)
gen12 = Conv2D(depth, kernel_size=1, padding='same', name='g12', data_format='channels_first')(gen11)
# gen13 = gen12[0,:,:,:]
# gen13 = reshape(gen13,[280,128,128])
```


```python
autoEncGen = Model(inputs=inputImg, outputs=gen12)
```


```python
autoEncGen.summary()
```

    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_img (InputLayer)          [(None, 1, 128, 128) 0                                            
    __________________________________________________________________________________________________
    conv1 (Conv2D)                  (None, 192, 64, 64)  3264        input_img[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 192, 64, 64)  256         conv1[0][0]                      
    __________________________________________________________________________________________________
    activation (Activation)         (None, 192, 64, 64)  0           batch_normalization[0][0]        
    __________________________________________________________________________________________________
    conv2 (Conv2D)                  (None, 192, 64, 64)  331968      activation[0][0]                 
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 192, 64, 64)  256         conv2[0][0]                      
    __________________________________________________________________________________________________
    add (Add)                       (None, 192, 64, 64)  0           activation[0][0]                 
                                                                     batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    activation_1 (Activation)       (None, 192, 64, 64)  0           add[0][0]                        
    __________________________________________________________________________________________________
    conv3 (Conv2D)                  (None, 384, 32, 32)  1180032     activation_1[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 384, 32, 32)  128         conv3[0][0]                      
    __________________________________________________________________________________________________
    activation_2 (Activation)       (None, 384, 32, 32)  0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    conv4 (Conv2D)                  (None, 384, 32, 32)  1327488     activation_2[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 384, 32, 32)  128         conv4[0][0]                      
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 384, 32, 32)  0           activation_2[0][0]               
                                                                     batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    activation_3 (Activation)       (None, 384, 32, 32)  0           add_1[0][0]                      
    __________________________________________________________________________________________________
    conv5 (Conv2D)                  (None, 768, 16, 16)  4719360     activation_3[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 768, 16, 16)  64          conv5[0][0]                      
    __________________________________________________________________________________________________
    activation_4 (Activation)       (None, 768, 16, 16)  0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    conv6 (Conv2D)                  (None, 768, 16, 16)  5309184     activation_4[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 768, 16, 16)  64          conv6[0][0]                      
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, 768, 16, 16)  0           activation_4[0][0]               
                                                                     batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, 768, 16, 16)  0           add_2[0][0]                      
    __________________________________________________________________________________________________
    conv7 (Conv2D)                  (None, 1536, 8, 8)   18875904    activation_5[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, 1536, 8, 8)   32          conv7[0][0]                      
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, 1536, 8, 8)   0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    conv8 (Conv2D)                  (None, 1536, 8, 8)   21235200    activation_6[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, 1536, 8, 8)   32          conv8[0][0]                      
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, 1536, 8, 8)   0           activation_6[0][0]               
                                                                     batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    activation_7 (Activation)       (None, 1536, 8, 8)   0           add_3[0][0]                      
    __________________________________________________________________________________________________
    conv9 (Conv2D)                  (None, 3072, 4, 4)   75500544    activation_7[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, 3072, 4, 4)   16          conv9[0][0]                      
    __________________________________________________________________________________________________
    activation_8 (Activation)       (None, 3072, 4, 4)   0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv10 (Conv2D)                 (None, 3072, 4, 4)   84937728    activation_8[0][0]               
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, 3072, 4, 4)   16          conv10[0][0]                     
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, 3072, 4, 4)   0           activation_8[0][0]               
                                                                     batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    activation_9 (Activation)       (None, 3072, 4, 4)   0           add_4[0][0]                      
    __________________________________________________________________________________________________
    trans1 (Conv2D)                 (None, 3072, 4, 4)   9440256     activation_9[0][0]               
    __________________________________________________________________________________________________
    reshape (Reshape)               (None, 1536, 2, 4, 4 0           trans1[0][0]                     
    __________________________________________________________________________________________________
    trans_out (Conv3D)              (None, 1536, 2, 4, 4 2360832     reshape[0][0]                    
    __________________________________________________________________________________________________
    gen1 (Conv3DTranspose)          (None, 768, 4, 8, 8) 75498240    trans_out[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, 768, 4, 8, 8) 32          gen1[0][0]                       
    __________________________________________________________________________________________________
    activation_10 (Activation)      (None, 768, 4, 8, 8) 0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    gen2 (Conv3DTranspose)          (None, 384, 8, 16, 1 18874752    activation_10[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, 384, 8, 16, 1 64          gen2[0][0]                       
    __________________________________________________________________________________________________
    activation_11 (Activation)      (None, 384, 8, 16, 1 0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    gen3 (Conv3DTranspose)          (None, 384, 8, 16, 1 3981696     activation_11[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, 384, 8, 16, 1 64          gen3[0][0]                       
    __________________________________________________________________________________________________
    activation_12 (Activation)      (None, 384, 8, 16, 1 0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    gen4 (Conv3DTranspose)          (None, 192, 16, 32,  4718784     activation_12[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, 192, 16, 32,  128         gen4[0][0]                       
    __________________________________________________________________________________________________
    activation_13 (Activation)      (None, 192, 16, 32,  0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    conv3d_transpose (Conv3DTranspo (None, 192, 16, 32,  995520      activation_13[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, 192, 16, 32,  128         conv3d_transpose[0][0]           
    __________________________________________________________________________________________________
    activation_14 (Activation)      (None, 192, 16, 32,  0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    gen6 (Conv3DTranspose)          (None, 96, 32, 64, 6 1179744     activation_14[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, 96, 32, 64, 6 256         gen6[0][0]                       
    __________________________________________________________________________________________________
    activation_15 (Activation)      (None, 96, 32, 64, 6 0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv3d_transpose_1 (Conv3DTrans (None, 96, 32, 64, 6 248928      activation_15[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, 96, 32, 64, 6 256         conv3d_transpose_1[0][0]         
    __________________________________________________________________________________________________
    activation_16 (Activation)      (None, 96, 32, 64, 6 0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    gen8 (Conv3DTranspose)          (None, 48, 64, 128,  294960      activation_16[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, 48, 64, 128,  512         gen8[0][0]                       
    __________________________________________________________________________________________________
    activation_17 (Activation)      (None, 48, 64, 128,  0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    conv3d_transpose_2 (Conv3DTrans (None, 48, 64, 128,  62256       activation_17[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, 48, 64, 128,  512         conv3d_transpose_2[0][0]         
    __________________________________________________________________________________________________
    activation_18 (Activation)      (None, 48, 64, 128,  0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    g10 (Conv3DTranspose)           (None, 1, 64, 128, 1 49          activation_18[0][0]              
    __________________________________________________________________________________________________
    tf_op_layer_Squeeze (TensorFlow [(None, 64, 128, 128 0           g10[0][0]                        
    __________________________________________________________________________________________________
    g12 (Conv2D)                    (None, 50, 128, 128) 3250        tf_op_layer_Squeeze[0][0]        
    ==================================================================================================
    Total params: 331,082,883
    Trainable params: 331,081,411
    Non-trainable params: 1,472
    __________________________________________________________________________________________________
    


```python
gen12
```




    <tf.Tensor 'g12/BiasAdd:0' shape=(None, 50, 128, 128) dtype=float32>



### Optimizer, checkpoints and model fitting
Here, we will define the loss function with other options like batch_size, no. of epochs, validation_split etc. We will also create checkpoints after certain no. of iteration. 

Create a custom loss function considering both `mse` and `psnr` <br>
** Uncomment if you want to use custom loss function


```python
# This loss function is for batch_size=1
'Functions used in metrics'
# @tf.function
def metrics_psnr(gt, pred):
    gt = tf.image.convert_image_dtype(gt, tf.float32)
    pred = tf.image.convert_image_dtype(pred, tf.float32)
    psnr_pred = tf.image.psnr(gt, pred, max_val = 1)

    return psnr_pred

# @tf.function
def metrics_mse(gt, pred):
    gt = tf.image.convert_image_dtype(gt, tf.float32)
    pred = tf.image.convert_image_dtype(pred, tf.float32)
    # mse_pred = tf.divide(tf.reduce_sum(tf.pow(tf.subtract(gt,pred),2.0)), tf.cast(tf.size(gt), tf.float32))
    mse_pred = tf.reduce_mean(tf.square(tf.subtract(gt, pred))) 

    return mse_pred  

'Loss function'
# @tf.function
def mse_psnr_loss(gt, pred):
    gt = tf.image.convert_image_dtype(gt, tf.float32)
    pred = tf.image.convert_image_dtype(pred, tf.float32)
    # mse_pred = tf.divide(tf.reduce_sum(tf.pow(tf.subtract(gt,pred),2.0)), tf.cast(tf.size(gt), tf.float32))
    mse_pred = tf.reduce_mean(tf.square(tf.subtract(gt, pred))) 

    psnr_pred = tf.image.psnr(gt, pred, max_val = 1)
    psnr_pred = psnr_pred/100
    psnr_pred = 1 - psnr_pred
    
    return mse_pred+psnr_pred

```


```python
op = tf.keras.optimizers.Adam(learning_rate = 0.001)
autoEncGen.compile(op,
                    loss='mean_squared_error',
                    metrics=['accuracy']
                    # metrics=['accuracy', metrics_psnr, metrics_mse]                   
                   )
```

Now, we will set the `learning rate` to reduce when there is no change in `val_loss` for a certain period.


```python
# Reducing learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=10,
                              min_lr=0.00001)
```

We will create `checkpoints` to store the model


```python
# Create checkpoint
checkpoint_path = "/content/drive/My Drive/3Dreconstruction/checkpoints2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(checkpoint_path,
                              monitor = 'val_loss',
                              verbose = 1,
                              save_best_only=False,
                              save_weights_only=False,
                              period=10)                            
```

    WARNING:tensorflow:`period` argument is deprecated. Please use `save_freq` to specify the frequency in number of batches seen.
    


```python
hist = autoEncGen.fit(x=X_train, 
                y=target, 
                batch_size=1,
                epochs=100,
                shuffle=True,
                validation_split = 0.2,
                #validation_data=(X_test, X_test)
                callbacks = [reduce_lr, cp_callback]
                )
```


```python
#-----> Uncomment to save model 
# autoEncGen.save("xRay-autoencoder-v10_7.model", save_format='tf') #save model
```


```python
#-----> Uncomment ot load model 
# model = load_model("xRay-autoencoder-v10_6.model",
#                    custom_objects={'mse_psnr_loss':mse_psnr_loss, 
#                                    'metrics_psnr': metrics_psnr,
#                                    'metrics_mse': metrics_mse}, ) #load model
```


```python
# encoded_imgs = encoder.predict(X_test)
# predicted = autoencoder.predict(X_test)
```

## Plotting
We will now plot training loss and validation loss. Then, we will save the figure in png format.


```python
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('mse_loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc = 'upper right')

plt.savefig('model_loss.png')
```

## Reference
This project is implemented based on the following paper - 
* Shen, Liyue, Wei Zhao, and Lei Xing. "Patient-specific reconstruction of volumetric computed tomography images from a single projection view via deep learning." Nature biomedical engineering 3.11 (2019): 880-888.
