# overview

## 标准卷积：2d

1. torch.nn.Conv2d
2. 参数是：**(kernel_num,input_channels,kernel_h,kernel_w)**（卷积核数量，输入通道数，卷积核高，卷积核宽）
3. 输入：**(batch_size,channels,higth,width)**针对的是2维张量图片，图片自身属性带有通道数。额外还有批量数。

![cnn过程](./../示例图片/cnn过程.gif)

![标准卷积](./../示例图片/标准卷积.png)

## 卷积维度：1d、2d、3d

卷积的维度并没有想象中复杂，一个普遍适用的理解：

1. 无论是1d、2d还是3d的卷积，**输入数据、输出数据以及卷积参数**的尺寸都要在**卷积维度**的基础上**+2**，比如：

   `nn.ConvXd`是X维卷积，卷积参数：(kernel_num,C~in~，\*X)，输入数据:（batch,C~in~,\*X）,输出数据:（batch,kernel_num,\*X）

2. 对于X维卷积`nn.ConvXd`，约束的其实是卷积核的尺寸，换句话说，**X维卷积`nn.ConvXd`的参数`kernel_size`必须是X维的**

3. 无论是1d、2d还是3d的卷积，卷积操作都是一样的：**X维度的卷积核在X维度数据上滑动计算**，只是，**输入数据、输出数据以及卷积参数**的尺寸都要在**卷积维度X**的基础上**+2**

- nn.Conv1d

  

## 分组卷积

# nn.Conv1d

# nn.Conv2d

```python
torch.nn.Conv2d(in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0, 
                dilation=1, 
                groups=1, 
                bias=True,
                padding_mode='zeros',
                device=None, 
                dtype=None
               )
```

- 参数

  1. `in_channels(int)`：表示输入数据的通道数$C_{in}$
  2. `out_channels`(int)：表示**卷积核（卷积滤波器）的数量**。每个卷积核都会生成一个输出特征，所以这也是卷积层输出的特征图数量图$C_{out}$
  3. `kernel_size(int or tuple)`：指定卷积核的大小，通常是**一个整数int或一个元组tuple (height, width)**。例如，`kernel_size=3` 表示一个 3x3 的卷积核。
  4. `stride(int or tuple)`：指定卷积操作的步幅，即卷积核在输入上滑动的步长。通常是一个整数或一个元组 (stride_height, stride_width)。
  5. `padding(int, tuple or str, optional)`：控制在卷积操作中是否要进行零填充。如果设置为0，表示不进行填充；如果设置为正整数，表示在输入周围填充相应数量的零。填充可以用来保持特征图尺寸不变，防止信息丢失。 **还可以是字符串类型：‘valid’、‘same’**，**valid表示不填充，same表示尺寸不变**
  6. `dilation`：指定卷积核的空洞（膨胀率）。这可以用来增大卷积核的感受野，以捕捉更大范围的特征。
  7. `groups`：用于分组卷积，通常将其设置为1，表示标准卷积操作。当设置为其他值时，卷积操作会被分成多个组，每个组内进行独立卷积，然后再组合。
  8. `bias`：一个布尔值，指定是否要在卷积层中使用偏置项。如果设置为True，将会有一个偏置项与每个卷积核相关联

- shape

  - input：$(N,C_{in},H_{in},W_{in})$或者$(C_{in},H_{in},W_{in})$

    其中N代表输入数据批量batch，$C_{in}$表示输入数据的通道数。$H_{in}$和$C_{in}$表示高和宽(行和列)。

    `torch.nn.Conv2d`会根据输入张量的维度来判断是是否包含批量batch。如果输入张量是四维，则认为带有batch。如果输入是三维张量，则认为输入数据是没有batch的单个样本。

  - output：$(N,C_{out},H_{out},W_{out})$或者$(C_{out},H_{out},W_{out})$

  - 必须：
    $$
    C_{in}=\text{in_channels}
    $$

    $$
    C_{out}=\text{out_channels}
    $$

    $$
    H_{out}=\frac{H_{in}+2*padding[0]-dilation[0]*(\text{kernel_size}[0]-1)-1}{stride[0]}+1
    $$

    $$
    W_{out}=\frac{W_{in}+2*padding[1]-dilation[1]*(\text{kernel_size}[1]-1)-1}{stride[1]}+1
    $$

    

# nn.Conv3d

# nn.ConvTransposed1d

# nn.ConvTranspose2d

# nn.ConvTranspose3d

# nn.LazyConv1d

# nn.LazyConv2d

# nn.LazyConv3d

# nn.LazyConvTranspose1d

# nn.LazyConvTranspose2d

# nn.LazyConvTranspose3d

# nn.Unfold

# nn.Fold