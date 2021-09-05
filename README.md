

# 🔥快速体验🔥

### 文件结构

```
|--MODEL               # 存放模型的文件夹
  |--Det_model         # 目标检测模型
  |--Seg_model         # 语义分割模型
|--TesImg              # 测试图片
|--OutImg              # 预测结果
|--tools               # 提供的图片预处理的工具
|--requirement.txt     # 依赖文件
|--predict.py          # 执行预测的脚本文件
```
### 环境准备

* #### 系统要求

  * ubuntu18.04

    直接可以使用，本教程的实测系统就是ubuntu18.04

  * ubuntu20.04

    因为ubuntu20.04没有预装gcc/g++编译环境故需要安装build-essential

    `sudo apt install build-essential`

 * #### python版本切换(也可以创建虚拟环境,更简单)
   
   * ubuntu18.04
   
   + 安装python3.7
   
   `sudo apt-get install python3.7`
   
   + 切换默认python为python3.7
   
   `sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 150`
   
 * #### 项目环境搭建
   + 导入依赖包

     `pip install -r requirement.txt`
     
   + 安装paddlex
     
     `pip install paddlex==2.0.0rc4 -i https://mirror.baidu.com/pypi/simple`
   
   + 安装paddle核心框架(这里默认安装的是cpu版本)
     
     `python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple`
   
     GPU版本参考<a href='https://www.paddlepaddle.org.cn/'> 这里</a>，其中cuda的版本可通过下面命令查看
     
     `nvidia-smi`
     
   + 加装一个依赖
   
     `pip install chardet`
   
 * #### 运行项目
   * python predict.py   