# Example processing with HookNet

You can find a concrete example of implementing a segmentation model in the processing.py file in this folder.
In this example, we use HookNet and change the processing loop accordingly to fit the requirements for HookNet. For more information about HookNet, please follow this link. Please note that your implementation probably requires different changes for other models and different settings. 

As mentioned, in this example, we use HookNet, which accepts an input shape of 1244x1244x3 for both the target and context branches. Both branches have a depth of 4, i.e., there are 4 pooling layers. Furthermore, HookNet uses 'valid padding,' resulting in an output shape of 1030x1030xC, where C represents the number of classes.







