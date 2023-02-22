# dynamic_videos
Motion tracking results of DeepMesh (the proposed method). 
Results are shown by overlaping meshes and images. Red contours are predicted results while green contours are ground truth
SAX | 2CH| 4CH 
--- | --- | --- 
<img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/sa_mid_pred.gif" width="120" height="120" /> | <img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/la_2ch_pred.gif" width="120" height="120" /> | <img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/la_4ch_pred.gif" width="120" height="120" /> 

Comparison methods
FFD | dDemons | 3D-UNet | MulViMotion|MeshMotion|DeepMesh(Proposed)
--- | --- | --- | ---| --- | ---
<img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/FFD_crop.gif" width="120" height="120" /> | <img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/dDemons_crop.gif" width="120" height="120" /> | <img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/3DUnet_crop.gif" width="120" height="120" /> | <img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/MulviMotion_crop.gif" width="120" height="120" /> | <img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/MeshMotion_crop.gif" width="120" height="120" /> | <img src="https://github.com/qmeng99/dynamic_videos/blob/main/gifs/DeepMesh_crop.gif" width="120" height="120" />


# Network Architectures

<p align="center">
    <img src="https://github.com/qmeng99/dynamic_videos/blob/main/network_architecture/DeformationNetwork.png" width="100%" height="100%">
</p>
**The network architecture of the Deformation network (H_D(·))**

<p align="center">
    <img src="https://github.com/qmeng99/dynamic_videos/blob/main/network_architecture/MotionNetwork.png" width="100%" height="100%">
</p>
**The network architecture of the Motion network (H_M(·))**

<p align="center">
    <img src="https://github.com/qmeng99/dynamic_videos/blob/main/network_architecture/2DFeatureLearning.png" width="100%" height="100%">
</p>
**The network architecture for 2D feature learning in the Motion network**


<p align="center">
    <img src="https://github.com/qmeng99/dynamic_videos/blob/main/network_architecture/3DFeatureLearning.png" width="100%" height="100%">
</p>
**The network architecture for 3D feature learning in the Motion network**


