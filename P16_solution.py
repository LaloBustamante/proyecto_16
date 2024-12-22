import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

# %%
def load_train(path):

    """
    It loads the train part of dataset from path
    """

    train_datagen = ImageDataGenerator(validation_split=0.25,
                                       rescale=1.0/255
                                )

    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=pd.read_csv(path + "labels.csv"),
        directory=path +'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16, # 32
        class_mode='raw',
        subset='training',
        seed=12345
    )

    return train_gen_flow

# %%
def load_test(path):

    """
    It loads the validation/test part of dataset from path
    """

    test_datagen = ImageDataGenerator(
        validation_split=0.25, rescale=1.0/255
    )

    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=pd.read_csv(path + "labels.csv"),
        directory=path +'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16, # 32
        class_mode='raw',
        subset='validation',
        seed=12345
    )

    return test_gen_flow

# %%
def create_model(input_shape=(224, 224, 3)):

    """
    It defines the model
    """

    backbone = ResNet50(
        input_shape=input_shape, 
        weights='imagenet',
        include_top=False
    )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.0001), # default is 0.001
        metrics=['mae']
    )

    #print(model.summary())
    return model

# %%
def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=20,
    steps_per_epoch=None,
    validation_steps=None
):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    """
    Trains the model given the parameters
    """

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2
    )

    return model

# %% [markdown]
# ## Prepare the Script to Run on the GPU Platform

# %% [markdown]
# Given you've defined the necessary functions you can compose a script for the GPU platform, download it via the "File|Open..." menu, and to upload it later for running on the GPU platform.
# 
# N.B.: The script should include the initialization section as well. An example of this is shown below.

# %%
# prepare a script to run on the GPU platform

init_str = """
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

import inspect

with open('run_model_on_gpu.py', 'w') as f:

    f.write(init_str)
    f.write('\n\n')

    for fn_name in [load_train, load_test, create_model, train_model]:

        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')

# %% [markdown]
# ### Output

# %% [markdown]
# Place the output from the GPU platform as an Markdown cell here.

# %% [markdown]
# 2023-04-10 04:41:07.285193: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
# 2023-04-10 04:41:07.338398: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
# Using TensorFlow backend.
# Found 5694 validated image filenames.
# Found 1897 validated image filenames.
# 2023-04-10 04:41:10.212230: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
# 2023-04-10 04:41:10.291251: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.291450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties:
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2023-04-10 04:41:10.291485: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2023-04-10 04:41:10.291529: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2023-04-10 04:41:10.344041: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2023-04-10 04:41:10.354479: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2023-04-10 04:41:10.459874: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2023-04-10 04:41:10.470656: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2023-04-10 04:41:10.470711: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2023-04-10 04:41:10.470806: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.471008: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.471142: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2023-04-10 04:41:10.471507: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 2023-04-10 04:41:10.494886: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300010000 Hz
# 2023-04-10 04:41:10.496657: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x3a2ff60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
# 2023-04-10 04:41:10.496684: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
# 2023-04-10 04:41:10.623815: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.624124: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x25127c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# 2023-04-10 04:41:10.624147: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
# 2023-04-10 04:41:10.624376: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.624565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties:
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2023-04-10 04:41:10.624616: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2023-04-10 04:41:10.624629: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2023-04-10 04:41:10.624662: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2023-04-10 04:41:10.624691: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2023-04-10 04:41:10.624721: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2023-04-10 04:41:10.624738: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2023-04-10 04:41:10.624747: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2023-04-10 04:41:10.624843: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.625084: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.625219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2023-04-10 04:41:10.626315: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2023-04-10 04:41:11.840376: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2023-04-10 04:41:11.840428: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0
# 2023-04-10 04:41:11.840438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N
# 2023-04-10 04:41:11.841791: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:11.842068: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:11.842246: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
# 2023-04-10 04:41:11.842289: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14988 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:1e.0, compute capability: 7.0)
# Downloading data from https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# 
#     8192/94765736 [..............................] - ETA: 1s
# 11714560/94765736 [==>...........................] - ETA: 0s
# 24510464/94765736 [======>.......................] - ETA: 0s
# 37552128/94765736 [==========>...................] - ETA: 0s
# 50290688/94765736 [==============>...............] - ETA: 0s
# 62988288/94765736 [==================>...........] - ETA: 0s
# 75726848/94765736 [======================>.......] - ETA: 0s
# 88399872/94765736 [==========================>...] - ETA: 0s
# 94773248/94765736 [==============================] - 0s 0us/step
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to
#   ['...']
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to
#   ['...']
# Train for 356 steps, validate for 119 steps
# Epoch 1/20
# 2023-04-10 04:41:26.034305: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2023-04-10 04:41:26.762342: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 356/356 - 54s - loss: 240.1997 - mae: 11.0967 - val_loss: 424.1841 - val_mae: 15.4917
# Epoch 2/20
# 356/356 - 38s - loss: 73.1405 - mae: 6.5176 - val_loss: 130.2860 - val_mae: 8.9525
# Epoch 3/20
# 356/356 - 37s - loss: 38.2911 - mae: 4.7705 - val_loss: 78.4883 - val_mae: 6.7900
# Epoch 4/20
# 356/356 - 37s - loss: 22.5524 - mae: 3.7080 - val_loss: 79.2300 - val_mae: 6.6938
# Epoch 5/20
# 356/356 - 38s - loss: 16.4091 - mae: 3.1380 - val_loss: 69.9737 - val_mae: 6.4156
# Epoch 6/20
# 356/356 - 38s - loss: 14.2627 - mae: 2.8796 - val_loss: 77.0521 - val_mae: 6.7016
# Epoch 7/20
# 356/356 - 38s - loss: 12.0682 - mae: 2.6552 - val_loss: 71.3144 - val_mae: 6.3349
# Epoch 8/20
# 356/356 - 38s - loss: 11.7980 - mae: 2.6073 - val_loss: 82.0825 - val_mae: 7.1355
# Epoch 9/20
# 356/356 - 38s - loss: 10.6117 - mae: 2.4818 - val_loss: 69.6299 - val_mae: 6.3353
# Epoch 10/20
# 356/356 - 38s - loss: 9.9028 - mae: 2.3691 - val_loss: 71.1683 - val_mae: 6.4408
# Epoch 11/20
# 356/356 - 38s - loss: 9.3474 - mae: 2.3169 - val_loss: 74.5620 - val_mae: 6.7784
# Epoch 12/20
# 356/356 - 37s - loss: 10.1380 - mae: 2.4163 - val_loss: 62.7982 - val_mae: 5.9394
# Epoch 13/20
# 356/356 - 38s - loss: 10.2378 - mae: 2.4257 - val_loss: 68.2272 - val_mae: 6.1635
# Epoch 14/20
# 356/356 - 38s - loss: 10.1878 - mae: 2.4146 - val_loss: 69.4524 - val_mae: 6.4428
# Epoch 15/20
# 356/356 - 38s - loss: 9.0975 - mae: 2.2912 - val_loss: 70.3128 - val_mae: 6.2446
# Epoch 16/20
# 356/356 - 37s - loss: 7.8139 - mae: 2.0939 - val_loss: 69.4005 - val_mae: 6.1634
# Epoch 17/20
# 356/356 - 38s - loss: 7.2549 - mae: 2.0230 - val_loss: 68.4559 - val_mae: 6.1004
# Epoch 18/20
# 356/356 - 37s - loss: 6.6996 - mae: 1.9719 - val_loss: 63.7187 - val_mae: 6.0363
# Epoch 19/20
# 356/356 - 38s - loss: 6.9101 - mae: 2.0044 - val_loss: 65.3447 - val_mae: 6.0575
# Epoch 20/20
# 356/356 - 38s - loss: 6.7118 - mae: 1.9685 - val_loss: 68.1251 - val_mae: 6.2608
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to
#   ['...']
# 119/119 - 9s - loss: 68.1251 - mae: 6.2608
# Test MAE: 6.2608
