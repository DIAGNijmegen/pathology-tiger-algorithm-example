# Example processing with HookNet

You can find a concrete example of implementing a segmentation model in the processing.py file in this folder.
In this example, we use HookNet and change the processing loop accordingly to fit the requirements for HookNet. For more information about HookNet, please follow this link. Please note that implementation probably requires different changes for other models and various settings. 

As mentioned, in this example, we use HookNet, which accepts an input shape of 1244x1244x3 for both the target and context branches. Both branches have a depth of 4, i.e., there are 4 pooling layers. Furthermore, HookNet uses 'valid padding,' resulting in an output shape of 1030x1030x7, where 7 represents the number of classes.

It makes sense to use a writing tile size of 1024 here, as this is a valid and efficient tile size for writing. Also, 1024 is divisible by 2^depth=2^4=16, which will also align the pooling over the whole slide.

Here we note that we have set the tile size to 1024:



Here we set the path to the HookNet config file


We also define a crop function which helps us center crop a numpy array is to a specific shape


