# 360 Degree Object Reconstruction using StructureSensor #

The fast global non-rigid registration algorithm for our 360 degree object reconstruction system.

# Related Publications #

Please cite this work if you make use of our system in any of your own endeavors:

* **[Online Global Non-rigid Registration for 3D Object Reconstruction Using Consumer-level Depth Cameras](http://www.cad.zju.edu.cn/home/weiweixu/wwxu2017_2018.files/2018_Online%20Global%20Non-rigid%20Registration%20for%203D%20Object%20Reconstruction.pdf)**
  **([video](https://youtu.be/SMli8-P7GJY))**
  , *Jiamin Xu, Weiwei Xu, Yin Yang, Zhigang Deng, Hujun Bao*, PG '18

|-- xObjReconCapture: iPad client program used to send the compressed StructureSensor RGBD images to server in real-time

|-- xServer: PC server 

|-- xObjRecon: Camera registration, surfel fusion, global non-rigid optimization algorithms

# Windows - Visual Studio #

* Windows 7/10 with Visual Studio 2015
* [OpenCV >= 3.2] (https://opencv.org/)
* [QT >= 5] (https://www.qt.io/)
* [OpenGL]
* [GLM]
* [CUDA >= 8.0] (https://developer.nvidia.com/cuda-downloads)
* [FLANN] (http://www.cs.ubc.ca/research/flann)
* [Eigen] (http://eigen.tuxfamily.org)