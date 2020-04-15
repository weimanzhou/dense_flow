This is a sample code for extrating dense flow field given a video.

Usage:
```bash
git clone url
cd dense_flow-master
g++ i_dense_flow.cpp -o i_dense_flow `pkg-config --cflags --libs opencv`
# 创建 source 文件夹，存放 avi 格式视频的位置
mkdir source
# 创建 target 文件夹，存放输出的默认位置
mkdir target
./i_dense_flow
# 执行后的 target 目录结构
# target
#   |- image
#   |- x_flow
#   |- y_flow
```

Exmaple:
```bash
$ tree source
source/
├── fil_cat2.avi
└── fil_cat.avi

0 directories, 2 files

$ ./i_dense_flow
$ tree target -L 2
$ tree target/ -L 2
target/
├── fil_cat
│   ├── image
│   ├── x_flow
│   └── y_flow
└── fil_cat2
    ├── image
    ├── x_flow
    └── y_flow

8 directories, 0 files
```
