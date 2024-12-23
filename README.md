# Yours_Colmap
使用自己的位姿数据、图像以及相应图像mask，进行稠密重建

1、Attention：

需要根据自己的项目要求编辑 {$workspace}/src/thirdparty/CMakeLists.txt 中关于poselib的cmake 信息

2、Adaption:

使用自己的位姿数据、图像以及相应的mask进行稠密重建，具体细节参考{$workspace}/src/colmap/elemrecond/*

3、Usage：

3.1 mkdir build && cd build 

3.2 cmake .. && make -j8

3.3 make install DESTDIR=/your/install/folder
