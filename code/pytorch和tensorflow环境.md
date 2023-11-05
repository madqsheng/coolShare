参考：https://www.tensorflow.org/install/gpu?hl=zh-cn

# 条件

深度学习代码（pytorch和tensorflow）可以在gpu上运行，有4个条件

1. **具备支持 NVIDIA GPU 的硬件**：计算机必须有支持 CUDA 的 NVIDIA GPU。不同的深度学习框架通常需要不同的计算能力（Compute Capability），请确保 GPU 显卡支持CUDA。[查阅支持 CUDA® 的 GPU 卡列表](https://developer.nvidia.com/cuda-gpus)
2. **安装 NVIDIA GPU 驱动程序**：安装适用的 GPU 的 NVIDIA 显卡驱动程序。确保驱动程序已正确安装和配置。[NVIDIA® GPU 驱动程序](https://www.nvidia.com/drivers)
3. **安装 CUDA 工具包**：深度学习框架通常需要与 CUDA 工具包兼容，因此需要安装合适版本的 CUDA 工具包。不同的框架可能对不同版本的 CUDA 有要求。[CUDA® 工具包](https://developer.nvidia.com/cuda-toolkit-archive)
4. **安装 cuDNN 库**：cuDNN 是 NVIDIA 提供的用于深度学习的 GPU 加速库。许多深度学习框架依赖于 cuDNN 进行加速，因此需要安装并配置正确版本的 cuDNN 库。[cuDNN SDK 8.1.0](https://developer.nvidia.com/cudnn)

注意：

1. 安装完了nvidia驱动程序以后可以在命令行输入：`nvidia-smi`，会出现类似：

   ![nvidia-smi](..\示例图片\nvidia-smi.png)

   明确一点：**右上角的CUDA version本并不是真正代表计算机中的cuda版本，甚至可能根本没有安装cuda**

2. **安装了cuda以后，命令行输入：`nvcc -V`会显示计算机真正的cuda版本**

3. **其实可以在电脑安装多个cuda版本，使用哪一个取决于环境变量的顺序**

# pytorch

很省心，**只有nvidia的驱动是一定要安装的，并且只需要注意python版本不要太旧即可**。

**pytorch的cuda和cudnn都是封在whl包里面，不依赖环境cuda版本。只依赖nvidia驱动版本，并且nv驱动新版都是兼容旧版cuda的**

1. 安装适合的nvidia驱动
2. 安装适合的python解释器版本
3. 安装pytorch：https://pytorch.org/

# tensorflow

兼容性问题很大

1. python版本还有要求3.6-3.9
2. **注意anaconda的版本不能太旧，否则`pip install tensorflow`都有可能报错**
3. 安装cuda，自动会添加到环境变量。
4. 还有cudnn也是必须的，解压到cuda的对应的位置