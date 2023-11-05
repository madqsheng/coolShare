# overview

哪些类型

# nn.MaxPool1d

# nn.MaxPool2d

```python
torch.nn.MaxPool2d(kernel_size,
                   stride=None, 
                   padding=0, 
                   dilation=1,
                   return_indices=False,
                   ceil_mode=False
                  )
```

- 参数

  1. `kernel_size(int or tuple)`：这是池化核的大小，它指定了每个池化窗口的尺寸。它通常是一个整数或一个元组 (height, width) 来指定高度和宽度上的窗口尺寸。
  2. `stride(int or tuple)`：指定了池化窗口在输入上滑动的步幅。它通常是一个整数或一个元组 (vertical_stride, horizontal_stride)。**默认是None，意思是默认等于kernel_size**。
  3. `padding(int or tuple)`：设置用于控制池化窗口如何处理输入边界的方式。**一个整数或元组**，用于手动指定填充的高度和宽度。注意：**和nn.Conv2d()不一样，nn.MaxPool2d不能设置str：'valid'（无填充）、'same'（使用零填充，保持输出大小不变）**
  4. `dilation(int or tuple)`：这是可选参数，用于指定在输入上的池化窗口内元素之间的空间间隔。默认值为 1。
  5. `return_indices(bool)`：一个布尔值，指示是否返回每个最大值的索引，以便在后续层中执行反池化操作。默认值为 False。
  6. `ceil_mode(bool)`：一个布尔值，当为True时，将使用**ceil(向上取整)**而不是**floor(向下取整)**来计算输出形状。默认为 False。

- shape

  - input：$(N,C,H_{in},W_{in})$或者$(C,H_{in},W_{in})$

    其中N代表输入数据批量batch，**运算前后的数据通道数是不变的**，$C$表示数据通道数。$H_{in}$和$C_{in}$表示高和宽(行和列)。

    `torch.nn.MaxPool2d`会根据输入张量的维度来判断是是否包含批量batch。如果输入张量是四维，则认为带有batch。如果输入是三维张量，则认为输入数据是没有batch的单个样本。

  - output：$(N,C,H_{out},W_{out})$或者$(C,H_{out},W_{out})$

  - 必须：
    $$
    H_{out}=\frac{H_{in}+2*padding[0]-dilation[0]*(\text{kernel_size}[0]-1)-1}{stride[0]}+1
    $$

    $$
    W_{out}=\frac{W_{in}+2*padding[1]-dilation[1]*(\text{kernel_size}[1]-1)-1}{stride[1]}+1
    $$

# nn.MaxPool3d

# nn.MaxUnpool1d

# nn.MaxUnpool2d

# nn.MaxUnpool3d

# nn.AvgPool1d

# nn.AvgPool2d

# nn.AvgPool3d

# nn.FractionalMaxPool2d

# nn.FractionalMaxPool3d

# nn.LPPool1d

# nn.LPPool2d

# nn.AdaptiveMaxPool1d

# nn.AdaptiveMaxPool2d

# nn.AdaptiveMaxPool3d

# nn.AdaptiveAvgPool1d

# nn.AdaptiveAvgPool2d

# nn.AdaptiveAvgPool3d