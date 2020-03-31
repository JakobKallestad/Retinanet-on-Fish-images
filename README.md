# Exercise 2 report

### By Jakob Kallestad, Eirik Norseng, Eirik Rikstad
---


```python
# About retinanet:
```

  

----

# Choice of network and infrastructure:

We decided to use a well known and well maintained implementation of RetinaNet object detection called Keras-Retinanet from https://github.com/fizyr/keras-retinanet/blob/master/README.md. This implementation is described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár (2017). More on this later.

  ---

  

# Prepare SVHN dataset:

First we needed to prepare the SVHN dataset. We downloaded the raw images from [ufldl.stanford.edu](http://ufldl.stanford.edu/housenumbers) and extracted the images to disk.


```python
import cv2
import tqdm
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
```

## Download SVHN Dataset:


```python
#!wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
#!wget http://ufldl.stanford.edu/housenumbers/test.tar.gz
#!wget http://ufldl.stanford.edu/housenumbers/extra.tar.gz
```

## Unzip the SVHN Dataset:


```python
#!tar -xf train.tar.gz
#!tar -xf test.tar.gz
#!tar -xf extra.tar.gz
```

## Preprocess annotation data:


```python
# Source: https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn?rq=1

# Get metadata from digitStruct.mat file
def get_box_data(index, hdf5_data):
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])
```


```python
# Put metadata into lists
train_pics = []
train_boxes = []
test_pics = []
test_boxes = []
```


```python
def metadata_to_lists(folder_name):
    pics = []
    boxes = []
    mat_data = h5py.File('{}/digitStruct.mat'.format(folder_name), 'r')
    size = mat_data['/digitStruct/name'].size
    for i in tqdm.tqdm(range(size)):
        pics.append((get_name(i, mat_data), folder_name))
        boxes.append(get_box_data(i, mat_data))
    return pics, boxes
```


```python
train_pics, train_boxes = metadata_to_lists('train')
extra_pics, extra_boxes = metadata_to_lists('extra')
test_pics, test_boxes = metadata_to_lists('test')

train_pics = train_pics + extra_pics
train_boxes = train_boxes + extra_boxes

print(len(train_pics))
print(len(test_pics))

train_pics, val_pics, train_boxes, val_boxes = train_test_split(train_pics, train_boxes, test_size=0.1, random_state=42)

print(len(train_pics))
print(len(val_pics))
print(len(test_pics))
```

      0%|          | 0/33402 [00:00<?, ?it/s]/usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:27: H5pyDeprecationWarning: dataset.value has been deprecated. Use dataset[()] instead.
    100%|██████████| 33402/33402 [03:07<00:00, 178.48it/s]
    100%|██████████| 202353/202353 [22:40<00:00, 148.70it/s]
    100%|██████████| 13068/13068 [01:09<00:00, 187.46it/s]


    235755
    13068
    212179
    23576
    13068


### Annotations format  

We converted the annotations to match the expected format of keras-retinanet which is explained on their [github](https://github.com/fizyr/keras-retinanet) as follows: 


`The CSV file with annotations should contain one annotation per line. Images with multiple bounding boxes should use one row per bounding box. Note that indexing for pixel values starts at 0. The expected format of each line is:`  
```python 
path/to/image.jpg,x1,y1,x2,y2,class_name
```  

source: [Keras-Retinanet on Github](https://github.com/fizyr/keras-retinanet)


```python
def create_annotation(pics, boxes):
    annotation = dict()
    annotation["1_img_path"] = []
    annotation["2_xmin"] = []
    annotation["3_ymin"] = []
    annotation["4_xmax"] = []
    annotation["5_ymax"] = []
    annotation["6_class_name"] = []

    for i in tqdm.tqdm(range(len(pics))):
        im = cv2.imread(pics[i][1]+'/'+pics[i][0])
        for j in range(len(boxes[i]['height'])):
            annotation["1_img_path"].append(pics[i][1]+'/'+pics[i][0])
            annotation["2_xmin"].append(min(im.shape[1]-1, max(0, int(boxes[i]['left'][j]))))
            annotation["3_ymin"].append(min(im.shape[0]-1, max(0, int(boxes[i]['top'][j]))))
            annotation["4_xmax"].append(min(im.shape[1], int(boxes[i]['left'][j] + boxes[i]['width'][j])))
            annotation["5_ymax"].append(min(im.shape[0], int(boxes[i]['top'][j] + boxes[i]['height'][j])))
            annotation["6_class_name"].append(int(boxes[i]['label'][j]))
    df = pd.DataFrame(annotation)
    return df
```


```python
df_train = create_annotation(train_pics, train_boxes)
df_validate = create_annotation(val_pics, val_boxes)
df_test = create_annotation(test_pics, test_boxes)
```

    100%|██████████| 212179/212179 [03:57<00:00, 893.33it/s] 
    100%|██████████| 23576/23576 [00:25<00:00, 909.77it/s]
    100%|██████████| 13068/13068 [00:18<00:00, 693.46it/s]


## Put annotation data into csv files


```python
df_train.to_csv('svhn_annotate_train.csv', header=None, index=None)
df_validate.to_csv('svhn_annotate_validate.csv', header=None, index=None)
df_test.to_csv('svhn_annotate_test.csv', header=None, index=None)
```


```python
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
with open('svhn_classes.csv', 'w') as file:
    for i, line in enumerate(classes):
        file.write('{},{}\n'.format(line,i))
```

  

----  

----  

  

# Prepare fish dataset:
We also prepared the FISH dataset in a similar way. We downloaded the zip file from [http://www.ii.uib.no/~ketil/fish-data.tgz](http://www.ii.uib.no/~ketil/fish-data.tgz) and extracted the images to disk.  

Then we genereated annotation csv files based on the txt files that corresponded to the images by their filename. These csv files also used the expected format of Keras-Retinanet which was explained earlier under __Annotations format__.


```python
import glob
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
```

## Download fish dataset:


```python
# Link to fish dataset (18GB!):
#!wget http://www.ii.uib.no/~ketil/fish-data.tgz
```

    --2020-03-18 20:26:15--  http://www.ii.uib.no/~ketil/fish-data.tgz
    Resolving www.ii.uib.no (www.ii.uib.no)... 129.177.16.249
    Connecting to www.ii.uib.no (www.ii.uib.no)|129.177.16.249|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 18827162822 (18G) [application/x-gzip]
    Saving to: ‘fish-data.tgz’
    
    fish-data.tgz       100%[===================>]  17.53G  17.6MB/s    in 29m 46s 
    
    2020-03-18 20:56:02 (10.1 MB/s) - ‘fish-data.tgz’ saved [18827162822/18827162822]
    


## Unzip fish dataset:


```python
#!tar -xzf fish-data.tgz
```

## Preprocess annotation data:


```python
def get_txt_file_paths(foldername):
    txt_file_paths = [filepath for filepath in glob.iglob('{}/*.txt'.format(foldername))]
    return txt_file_paths
```


```python
txt_file_paths = get_txt_file_paths('img_sim_train_2017_4_species')
print(len(txt_file_paths))
txt_file_paths += get_txt_file_paths('img_sim_train_2018_4_species')
print(len(txt_file_paths))
```

    10000
    20000



```python
txt_train_paths, txt_test_paths = train_test_split(txt_file_paths, test_size=0.1, random_state=42)
txt_train_paths, txt_val_paths = train_test_split(txt_train_paths, test_size=0.1, random_state=42)
print("Number of images in training set: ", len(txt_train_paths))
print("Number of images in validation set: ", len(txt_val_paths))
print("Number of images in testing set: ", len(txt_test_paths))
```

    Number of images in training set:  16200
    Number of images in validation set:  1800
    Number of images in testing set:  2000



```python
def txt_annotations_to_csv(txt_paths):
    master_df = pd.DataFrame()
    for path in tqdm.tqdm(txt_paths):
        df = pd.read_csv(path, header = None)
        df[0] = df[0].apply(lambda x: x[45:])
        master_df = master_df.append(df, ignore_index = True)
    return master_df
```


```python
df_train = txt_annotations_to_csv(txt_train_paths)
df_val = txt_annotations_to_csv(txt_val_paths)
df_test = txt_annotations_to_csv(txt_test_paths)
classes = df_train[5].unique()

print(len(df_train))
print(len(df_val))
print(len(df_test))
print(classes)
```

    100%|██████████| 16200/16200 [02:05<00:00, 128.79it/s]
    100%|██████████| 1800/1800 [00:08<00:00, 210.65it/s]
    100%|██████████| 2000/2000 [00:09<00:00, 207.54it/s]

    89203
    9808
    10997
    ['bluewhiting' 'mackerel' 'benthosema' 'herring']


    


## Put annotation data into csv files:


```python
df_train.to_csv('fish_annotate_train.csv', header=None, index=None)
df_val.to_csv('fish_annotate_validate.csv', header=None, index=None)
df_test.to_csv('fish_annotate_test.csv', header=None, index=None)
```


```python
with open('fish_classes.csv', 'w') as file:
    for i, line in enumerate(sorted(classes)):
        file.write('{},{}\n'.format(line,i))
```

----

----

## Installing Keras-Retinanet

To install Keras-RetinaNet we ran the following bit of code:


```python
!git clone https://github.com/fizyr/keras-retinanet/
%cd keras-retinanet/
!pip install .
!python3 setup.py build_ext --inplace
%cd ..
print("done")
```

    done


## Training:

For training there is a special command from Keras-RetinaNet that we ran inside a terminal screen window. We started with a resnet50 model pre-trained on the COCO dataset which we downloaded from [here](https://github.com/fizyr/keras-retinanet/releases) For learning rate we used the default of keras-retinanet which starts at 1e-5 and decreases automatically if the loss does not decrease for a few epochs. The final learning rate ended up at 1e-8 for the fish model where we decided to stop the training because the validation accuracy didn't seem to increase much more.  
  
The training procedures took about 1-2 hours per epoch on a Tesla V100 GPU, and we ended up training for 20 epochs for the SVHN model and 10 epochs for the FISH model.  
  
__For SVHN images:__
```python
!retinanet-train \
--weights _pretrained_model.h5 \
--initial-epoch 0 \
--epochs 100 \
--steps 10000 \
--batch-size 4 \
--snapshot-path './svhn_snapshots/' \
--weighted-average \
csv svhn_annotate_train.csv svhn_classes.csv \
--val-annotations svhn_annotate_validate.csv
```

__For FISH images:__
```python
!retinanet-train \
--weights _pretrained_model.h5 \
--initial-epoch 0 \
--epochs 100 \
--steps 10000 \
--batch-size 4 \
--snapshot-path './fish_snapshots/' \
--weighted-average \
csv fish_annotate_train.csv fish_classes.csv \
--val-annotations fish_annotate_validate.csv
```

## Testing:

First we found the path to the most recent SVHN and FISH models.


```python
import os
print(os.path.join('svhn_snapshots', sorted(os.listdir('svhn_snapshots'), reverse=True)[0]))
print(os.path.join('fish_snapshots', sorted(os.listdir('fish_snapshots'), reverse=True)[0]))
```

    svhn_snapshots/resnet50_csv_21.h5
    fish_snapshots/resnet50_csv_10.h5


  

Then we ran the following commands in terminal windows to meassure the models performance on the test data:


```python
!retinanet-evaluate \
--convert-model \
csv svhn_annotate_test.csv svhn_classes.csv \
svhn_snapshots/resnet50_csv_21.h5
```

    Using TensorFlow backend.
    2020-03-31 18:24:00.495280: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
    2020-03-31 18:24:00.497176: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
    Loading model, this may take a second...
    2020-03-31 18:24:02.546934: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2020-03-31 18:24:03.230676: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.231414: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-03-31 18:24:03.231477: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-03-31 18:24:03.231521: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-03-31 18:24:03.233490: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-03-31 18:24:03.233856: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-03-31 18:24:03.235922: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-03-31 18:24:03.237080: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-03-31 18:24:03.237148: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-03-31 18:24:03.237261: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.237978: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.238613: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
    2020-03-31 18:24:03.238922: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
    2020-03-31 18:24:03.247288: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2000129999 Hz
    2020-03-31 18:24:03.247771: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5612c678b350 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-03-31 18:24:03.247797: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-03-31 18:24:03.337451: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.338220: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5612c67ae480 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-03-31 18:24:03.338269: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
    2020-03-31 18:24:03.338538: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.339199: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-03-31 18:24:03.339246: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-03-31 18:24:03.339263: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-03-31 18:24:03.339299: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-03-31 18:24:03.339344: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-03-31 18:24:03.339358: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-03-31 18:24:03.339374: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-03-31 18:24:03.339388: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-03-31 18:24:03.339462: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.340126: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.340789: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
    2020-03-31 18:24:03.340887: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-03-31 18:24:03.755014: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-03-31 18:24:03.755072: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
    2020-03-31 18:24:03.755082: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
    2020-03-31 18:24:03.755385: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.756116: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:24:03.756767: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14950 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[-22.627417, -11.313708,  22.627417,  11.313708],
           [-28.50876 , -14.25438 ,  28.50876 ,  14.25438 ],
           [-35.918785, -17.959393,  35.918785,  17.959393],
           [-16.      , -16.      ,  16.      ,  16.      ],
           [-20.158737, -20.158737,  20.158737,  20.158737],
           [-25.398417, -25.398417,  25.398417,  25.398417],
           [-11.313708, -22.627417,  11.313708,  22.627417],
           [-14.25438 , -28.50876 ,  14.25438 ,  28.50876 ],
           [-17.959393, -35.918785,  17.959393,  35.918785]], dtype=float32)> anchors
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[-45.254833, -22.627417,  45.254833,  22.627417],
           [-57.01752 , -28.50876 ,  57.01752 ,  28.50876 ],
           [-71.83757 , -35.918785,  71.83757 ,  35.918785],
           [-32.      , -32.      ,  32.      ,  32.      ],
           [-40.317474, -40.317474,  40.317474,  40.317474],
           [-50.796833, -50.796833,  50.796833,  50.796833],
           [-22.627417, -45.254833,  22.627417,  45.254833],
           [-28.50876 , -57.01752 ,  28.50876 ,  57.01752 ],
           [-35.918785, -71.83757 ,  35.918785,  71.83757 ]], dtype=float32)> anchors
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[ -90.50967 ,  -45.254833,   90.50967 ,   45.254833],
           [-114.03504 ,  -57.01752 ,  114.03504 ,   57.01752 ],
           [-143.67514 ,  -71.83757 ,  143.67514 ,   71.83757 ],
           [ -64.      ,  -64.      ,   64.      ,   64.      ],
           [ -80.63495 ,  -80.63495 ,   80.63495 ,   80.63495 ],
           [-101.593666, -101.593666,  101.593666,  101.593666],
           [ -45.254833,  -90.50967 ,   45.254833,   90.50967 ],
           [ -57.01752 , -114.03504 ,   57.01752 ,  114.03504 ],
           [ -71.83757 , -143.67514 ,   71.83757 ,  143.67514 ]],
          dtype=float32)> anchors
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[-181.01933,  -90.50967,  181.01933,   90.50967],
           [-228.07008, -114.03504,  228.07008,  114.03504],
           [-287.35028, -143.67514,  287.35028,  143.67514],
           [-128.     , -128.     ,  128.     ,  128.     ],
           [-161.2699 , -161.2699 ,  161.2699 ,  161.2699 ],
           [-203.18733, -203.18733,  203.18733,  203.18733],
           [ -90.50967, -181.01933,   90.50967,  181.01933],
           [-114.03504, -228.07008,  114.03504,  228.07008],
           [-143.67514, -287.35028,  143.67514,  287.35028]], dtype=float32)> anchors
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[-362.03867, -181.01933,  362.03867,  181.01933],
           [-456.14017, -228.07008,  456.14017,  228.07008],
           [-574.70056, -287.35028,  574.70056,  287.35028],
           [-256.     , -256.     ,  256.     ,  256.     ],
           [-322.5398 , -322.5398 ,  322.5398 ,  322.5398 ],
           [-406.37466, -406.37466,  406.37466,  406.37466],
           [-181.01933, -362.03867,  181.01933,  362.03867],
           [-228.07008, -456.14017,  228.07008,  456.14017],
           [-287.35028, -574.70056,  287.35028,  574.70056]], dtype=float32)> anchors
    Running network: N/A% (0 of 13068) |     | Elapsed Time: 0:00:00 ETA:  --:--:--2020-03-31 18:24:17.723064: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-03-31 18:24:19.728236: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    Running network: 100% (13068 of 13068) |#| Elapsed Time: 0:10:36 Time:  0:10:36
    Parsing annotations: 100% (13068 of 13068) || Elapsed Time: 0:00:00 Time:  0:00:00
    5099 instances of class 1 with average precision: 0.8652
    4149 instances of class 2 with average precision: 0.9013
    2882 instances of class 3 with average precision: 0.8461
    2523 instances of class 4 with average precision: 0.8785
    2384 instances of class 5 with average precision: 0.8760
    1977 instances of class 6 with average precision: 0.8600
    2019 instances of class 7 with average precision: 0.8892
    1660 instances of class 8 with average precision: 0.8689
    1595 instances of class 9 with average precision: 0.8652
    1744 instances of class 10 with average precision: 0.8876
    Inference time for 13068 images: 0.0459
    mAP using the weighted average of precisions among classes: 0.8743
    mAP: 0.8738



```python
!retinanet-evaluate \
--convert-model \
csv fish_annotate_test.csv fish_classes.csv \
fish_snapshots/resnet50_csv_10.h5
```

    Using TensorFlow backend.
    2020-03-31 18:35:34.133599: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
    2020-03-31 18:35:34.135282: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
    Loading model, this may take a second...
    2020-03-31 18:35:35.376617: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2020-03-31 18:35:36.054544: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.055246: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-03-31 18:35:36.055302: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-03-31 18:35:36.055347: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-03-31 18:35:36.057229: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-03-31 18:35:36.057609: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-03-31 18:35:36.059533: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-03-31 18:35:36.060682: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-03-31 18:35:36.060746: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-03-31 18:35:36.060845: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.061538: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.062171: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
    2020-03-31 18:35:36.062492: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
    2020-03-31 18:35:36.070154: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2000129999 Hz
    2020-03-31 18:35:36.070852: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ba867a0de0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2020-03-31 18:35:36.070879: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    2020-03-31 18:35:36.160783: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.161588: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ba86827380 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
    2020-03-31 18:35:36.161628: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
    2020-03-31 18:35:36.161861: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.162611: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
    pciBusID: 0000:00:04.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
    coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.75GiB deviceMemoryBandwidth: 836.37GiB/s
    2020-03-31 18:35:36.162693: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-03-31 18:35:36.162724: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    2020-03-31 18:35:36.162765: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
    2020-03-31 18:35:36.162786: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
    2020-03-31 18:35:36.162805: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
    2020-03-31 18:35:36.162823: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
    2020-03-31 18:35:36.162838: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-03-31 18:35:36.162943: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.163784: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.164504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
    2020-03-31 18:35:36.164677: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
    2020-03-31 18:35:36.581236: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
    2020-03-31 18:35:36.581291: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0 
    2020-03-31 18:35:36.581301: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N 
    2020-03-31 18:35:36.581569: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.582431: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2020-03-31 18:35:36.583160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14950 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0)
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[-22.627417, -11.313708,  22.627417,  11.313708],
           [-28.50876 , -14.25438 ,  28.50876 ,  14.25438 ],
           [-35.918785, -17.959393,  35.918785,  17.959393],
           [-16.      , -16.      ,  16.      ,  16.      ],
           [-20.158737, -20.158737,  20.158737,  20.158737],
           [-25.398417, -25.398417,  25.398417,  25.398417],
           [-11.313708, -22.627417,  11.313708,  22.627417],
           [-14.25438 , -28.50876 ,  14.25438 ,  28.50876 ],
           [-17.959393, -35.918785,  17.959393,  35.918785]], dtype=float32)> anchors
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[-45.254833, -22.627417,  45.254833,  22.627417],
           [-57.01752 , -28.50876 ,  57.01752 ,  28.50876 ],
           [-71.83757 , -35.918785,  71.83757 ,  35.918785],
           [-32.      , -32.      ,  32.      ,  32.      ],
           [-40.317474, -40.317474,  40.317474,  40.317474],
           [-50.796833, -50.796833,  50.796833,  50.796833],
           [-22.627417, -45.254833,  22.627417,  45.254833],
           [-28.50876 , -57.01752 ,  28.50876 ,  57.01752 ],
           [-35.918785, -71.83757 ,  35.918785,  71.83757 ]], dtype=float32)> anchors
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[ -90.50967 ,  -45.254833,   90.50967 ,   45.254833],
           [-114.03504 ,  -57.01752 ,  114.03504 ,   57.01752 ],
           [-143.67514 ,  -71.83757 ,  143.67514 ,   71.83757 ],
           [ -64.      ,  -64.      ,   64.      ,   64.      ],
           [ -80.63495 ,  -80.63495 ,   80.63495 ,   80.63495 ],
           [-101.593666, -101.593666,  101.593666,  101.593666],
           [ -45.254833,  -90.50967 ,   45.254833,   90.50967 ],
           [ -57.01752 , -114.03504 ,   57.01752 ,  114.03504 ],
           [ -71.83757 , -143.67514 ,   71.83757 ,  143.67514 ]],
          dtype=float32)> anchors
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[-181.01933,  -90.50967,  181.01933,   90.50967],
           [-228.07008, -114.03504,  228.07008,  114.03504],
           [-287.35028, -143.67514,  287.35028,  143.67514],
           [-128.     , -128.     ,  128.     ,  128.     ],
           [-161.2699 , -161.2699 ,  161.2699 ,  161.2699 ],
           [-203.18733, -203.18733,  203.18733,  203.18733],
           [ -90.50967, -181.01933,   90.50967,  181.01933],
           [-114.03504, -228.07008,  114.03504,  228.07008],
           [-143.67514, -287.35028,  143.67514,  287.35028]], dtype=float32)> anchors
    tracking <tf.Variable 'Variable:0' shape=(9, 4) dtype=float32, numpy=
    array([[-362.03867, -181.01933,  362.03867,  181.01933],
           [-456.14017, -228.07008,  456.14017,  228.07008],
           [-574.70056, -287.35028,  574.70056,  287.35028],
           [-256.     , -256.     ,  256.     ,  256.     ],
           [-322.5398 , -322.5398 ,  322.5398 ,  322.5398 ],
           [-406.37466, -406.37466,  406.37466,  406.37466],
           [-181.01933, -362.03867,  181.01933,  362.03867],
           [-228.07008, -456.14017,  228.07008,  456.14017],
           [-287.35028, -574.70056,  287.35028,  574.70056]], dtype=float32)> anchors
    Running network: N/A% (0 of 2000) |      | Elapsed Time: 0:00:00 ETA:  --:--:--2020-03-31 18:35:50.096073: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2020-03-31 18:35:52.094416: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
    Running network: 100% (2000 of 2000) |###| Elapsed Time: 0:04:27 Time:  0:04:27
    Parsing annotations: 100% (2000 of 2000) || Elapsed Time: 0:00:00 Time:  0:00:00
    2807 instances of class benthosema with average precision: 0.9396
    2699 instances of class bluewhiting with average precision: 0.9302
    2784 instances of class herring with average precision: 0.9249
    2707 instances of class mackerel with average precision: 0.9205
    Inference time for 2000 images: 0.0442
    mAP using the weighted average of precisions among classes: 0.9289
    mAP: 0.9288


### Results:

mAP for SVHN = __0.87__  
mAP for FISH = __0.93__  

  

## Output Images:

For outputing examples of images with the models predicted bounding boxes and the ground truth bounding boxes we ran retinanet-evalute again, but with --save-path argument and also --score-threshold and --max-detections for nicer looking images.

__For SVHN images:__
```python
!retinanet-evaluate \
--convert-model \
--score-threshold 0.5 \
--max-detections 4 \
--save-path svhn_predicted_images \
csv svhn_annotate_test.csv svhn_classes.csv \
svhn_snapshots/resnet50_csv_21.h5
```

__For FISH images:__
```python
!retinanet-evaluate \
--convert-model \
--score-threshold 0.6 \
--save-path fish_predicted_images \
csv fish_annotate_test.csv fish_classes.csv \
fish_snapshots/resnet50_csv_10.h5
```

  

## Lets take a look at some of the output images:


```python
svhn_images = [505, 592, 720, 782, 830, 835, 900, 1002, 1004, 1007]
fish_images = [0, 1, 3, 100, 1000, 1001, 1005, 1010, 1015, 1020]
```


```python
import matplotlib.pyplot as plt
import cv2

for img_dir, img_inds in zip(('svhn_predicted_images', 'fish_predicted_images'), (svhn_images, fish_images)):
    biggest_images = sorted(os.listdir(img_dir), key=lambda filename: -os.path.getsize(os.path.join(img_dir, filename)))
    for ind in img_inds:
        img = cv2.imread(img_dir + '/' + biggest_images[ind])
        plt.figure(figsize=(50, 50))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
```


![png](output_65_0.png)



![png](output_65_1.png)



![png](output_65_2.png)



![png](output_65_3.png)



![png](output_65_4.png)



![png](output_65_5.png)



![png](output_65_6.png)



![png](output_65_7.png)



![png](output_65_8.png)



![png](output_65_9.png)



![png](output_65_10.png)



![png](output_65_11.png)



![png](output_65_12.png)



![png](output_65_13.png)



![png](output_65_14.png)



![png](output_65_15.png)



![png](output_65_16.png)



![png](output_65_17.png)



![png](output_65_18.png)



![png](output_65_19.png)


  
  

# Conclusion:

The model performed well on both SVHN and FISH images. Especially for fish images we saw examples above of the model being able to handle overlapping and partially unobservable objects with relative ease. It was also easy to implement and did not require us to specify and bounding boxes or much fine tuning at all.

# Future work:

In order to improve the accuracy further it could be an idea to swap out the ResNet50 backbone with something bigger like ResNet152. Another potentially helpfull idea for SVHN digit classification would be to transform the SVHN images to grayscale images by scaling the RBG values of each image by a factor of (0.2989, 0.5870, 0.1140)
