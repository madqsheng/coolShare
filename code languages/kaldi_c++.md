# 1. 概述

理解：

- 开源的语音识别，C++工具包，高度模块化的设计
- 包含了许多先进的语音处理算法和技术，如**声学模型**（基于高斯混合模型和深度神经网络）、**语言模型、特征提取、解码器**等。
- Kaldi还提供了丰富的工具和脚本，用于**数据准备、模型训练、评估和调优**等任务
- Kaldi的主要目标是提供一个灵活、可扩展且高效的框架，用于**构建和训练**各种语音识别系统，包括**自动语音识别（ASR）、说话人识别、语音合成**等应用。

# 2. 安装kaldi

## 2.1 linux系统

ubuntu系统

参考：

https://cloud.tencent.com/developer/article/1721926

[语音识别--kaldi环境搭建（基于Ubuntu系统） - 掘金 (juejin.cn)](https://juejin.cn/post/6970225607669973029)

步骤：

1. 下载依赖项：

   ```bash
   sudo apt-get update
   sudo apt-get install git subversion sox automake autoconf g++ python python3
   sudo apt-get install zlib1g-dev libtool libatlas-base-dev libatlas3-base
   ```

   大概率是补全的，在后续的编译中可能会报错，需要继续安装

   以确保您的软件包索引是最新的：

   ```bash
   sudo apt update
   ```

2. 下载kaldi：

   ```bash
   git clone https://github.com/kaldi-asr/kaldi.git
   ```

   说明：

   - **tools**目录全是kaldi依赖的包，
   - **egs**为kaldi提供的实例，其中包含常用的数据集运行的源码
   - **src**目录为kaldi 的源代码

3. 检测依赖项：

   ```bash
   cd kaldi/tools/extras/
   ./check_dependencies.sh 
   ```

   

4. 进入tools目录下进行编译

   ```bash
   cd kaldi/tools
   # make
   # (多核并行) 下载编译
   make -j 4
   ```

5. 进入src目录进行配置并编译

   ```bash
   cd ../src
   
   # 查看执行步骤
   cat INSTALL
   
   # kaldi运行前配置
   ./configure --shared
   
   # kaldi编译
   make depend -j 4
   make -j 4
   ```

   ./configure 报错：**UserCould not find the MKL library directory.**

   make过程中比较好费时间，当日志最后显示为

   　　　　　　Done

   就成功了。

6. 测试：

   ```bash
   cd ../egs/yesno/s5
   ./run.sh
   ```

   

## 2.2 windows系统

方法：

1. 通过使用**Windows的Linux子系统（WSL）**来在Windows上安装Kaldi
2. **使用Docker**：你可以使用Docker来运行Kaldi容器。你可以从Docker Hub上找到适用于Kaldi的预构建Docker镜像，并在Windows上安装Docker Desktop，然后在容器中运行Kaldi。
3. **使用Windows下的Cygwin**：Cygwin是一个在Windows上提供类Unix环境的开源工具。你可以通过安装Cygwin来在Windows上模拟Linux环境，并尝试按照Linux下的安装步骤来安装Kaldi。

# 3. 工作流

1. 设置环境变量

   - 如果你是通过包管理器（比如apt）来安装Kaldi的，通常Kaldi会被安装在系统的默认位置。`/usr/share/kaldi`或者`/usr/local/share/kaldi`等位置
   - 通过源代码编译安装Kaldi的，那么Kaldi会被安装在你指定的路径下。比如我的`~/桌面/kaldi/src/bin`

   ```bash
   sudo vim ~/.bashrc
   # 输入
   export KALDI_ROOT=~/桌面/kaldi
   export PATH=$KALDI_ROOT/src/bin:$PATH
   
   source ~/.bashrc
   ```

   注意：

   如果你不设置`KALDI_ROOT`环境变量，仍然可以在C++项目中调用Kaldi的库文件，但是**需要手动指定Kaldi库文件的路径**。通常情况下，Kaldi的库文件位于`$KALDI_ROOT/src/lib`目录下。

   在创建C++项目时，需要确保正确链接Kaldi的库文件。

   假设Kaldi安装在`/path/to/kaldi`目录下，**Kaldi的库文件**于`/path/to/kaldi/src/lib`目录下，可以按照以下步骤来编译并链接你的C++项目：

2. 

# 4. 学习

参考：

https://zhuanlan.zhihu.com/p/444867152

https://blog.csdn.net/wbgxx333/article/details/38962623