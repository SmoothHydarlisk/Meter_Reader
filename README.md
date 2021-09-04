

# 🔥快速体验🔥

### 文件结构

```
|--MODEL               # 存放模型的文件夹
  |--Det_model         # 目标检测模型
  |--Seg_model         # 语义分割模型
|--TesImg              # 测试图片
|--OutImg              # 预测结果
|--requirement.txt     # 依赖文件
|--predict.py          # 执行预测的脚本文件
```
### 环境准备

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
   
   + 安装paddle核心框架
     
     `python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple`
   
   + 加装一个依赖
   
     `pip install chardet`
   
 * #### 运行项目
   * python predict.py   