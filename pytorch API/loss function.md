# overview

## 交叉熵

$$
L_{Cross Entropy}(P(\pmb{x}),Q(\pmb{x})) = \sum_{i=1}^{n}q(x_i)log(p(x_i))
$$

度量两个分布$P(\pmb{x})$和$Q(\pmb{x})$的相似度，$\pmb{x}$其实是一个随机变量

交叉熵演变的花样很多，本质其实要看随机变量$\pmb{x}$的取值：$x_1,...,x_n$

对于交叉熵最关键的要点：**一个样本要包括所有的取值**，输入数据的样本在哪一维度。

### 二元交叉熵Binary Cross Entropy

只有两个分类，所以本质是：$L_{Binary Cross Entropy}(P(\pmb{x}),Q(\pmb{x})) = q(x_1)log(p(x_1))+q(x_2)log(p(x_2))$

pytorch提供的相关接口：

1. `nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')(input, target)`
2. `nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)(input, target)`

输入：`input`和`target`

要点：

1. 输入数据`input`和`target`的每个元素互相对应，**一个元素算做一个样本**：计算两种可能，一是元素的取值$input[i]$和$target[i]$，二是元素取反$(1-input[i])$和$(1-target[i])$，只需要提供数据，接口API也会自动计算对应元素取反计算交叉熵
2. 输入数据`input`和`target`形状必须相同
3. 前面的取反操作要求：输入数据`input`和`target`的元素取值范围在[0,1]之间
4. `nn.BCELoss`和`nn.BCEWithLogitsLoss`唯一不同的地方在于，`nn.BCEWithLogitsLoss`会在内部对`input`进行`sigmoid`操作（元素自己）。要求输入数据元素取值范围在[0,1]之间是固有要求。内部自带的`sigmoid`函数可以让`input`满足这一点。换言之：`nn.BCELoss`要求输入数据`input`和`target`的元素取值范围在[0,1]之间。`nn.BCEWithLogitsLoss`只要求`target`的元素取值范围在[0,1]之间，`input`没有这样的要求，因为内部的`sigmoid`可以完成。

### 多分类交叉熵

对于输入数据，**也许一行、一列甚至一个矩阵构成一个样本**。样本里元素数目代表有多少类。元素的取值是相应类的概率。

所以最关键的是，**究竟一个样本是哪些数据**？
$$
L_{Cross Entropy}(P(\pmb{x}),Q(\pmb{x})) = \sum_{i=1}^{n}q(x_i)log(p(x_i))
$$
pytorch提供的相关接口：

`torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)(input, target)`

对于这个API接口，究竟一个样本是哪些数据呢？有3个要点：

1. 直接给出答案，无论输入数据`input`和`target`是多少维的数据，形状是什么。都是将它们**第二维里所有元素作为一个样本(dim=1)**

2. 内部对`input`在**第二维度**上执行`softmax`操作(dim=1)

3. 针对输入数据`input`和`target`维度不一样的情况，设计了方便之处：

   特别用法：**`target`比`input`少一维度**

   之所以`target`比`input`少一维度，是因为`target`里每个元素取值不是概率值，而是**类索引**

   **这个API定死了第二维度所有元素当一个样本**。所以当**`target`比`input`少一维度**这种情况，会在第二维度上进行onehot操作，之后的操作就是上面的操作1和2

# nn.L1Loss

# nn.MSELoss

# nn.CrossEntropyLoss

**多分类交叉熵损失函数**

3个要点：

1. 直接给出答案，无论输入数据`input`和`target`是多少维的数据，形状是什么。都是将它们**第二维里所有元素作为一个样本(dim=1)**

2. 内部对`input`在**第二维度**上执行`softmax`操作(dim=1)

3. 针对输入数据`input`和`target`维度不一样的情况，设计了方便之处：

   特别用法：**`target`比`input`少一维度**

   之所以`target`比`input`少一维度，是因为`target`里每个元素取值不是概率值，而是**类索引**

   **这个API定死了第二维度所有元素当一个样本**。所以当**`target`比`input`少一维度**这种情况，会在第二维度上进行onehot操作，之后的操作就是上面的操作1和2

```python
torch.nn.CrossEntropyLoss(weight=None, 
                          size_average=None, 
                          ignore_index=-100, 
                          reduce=None, 
                          reduction='mean', 
                          label_smoothing=0.0
                         )(input,target)
```

最重要掌握：**这个API定死了第二维度所有元素当一个样本，无论input和target的维度和形状怎么样**

还原计算过程

```python
import torch.nn as nn
import torch
import torch.nn.functional as F

# input, target形同形状
loss = nn.CrossEntropyLoss(reduction='none')
input = torch.randn(2,3,5)
target = torch.randn(2,3,5)
output = loss(input, target)
print(output)

input_softmax=F.softmax(input,dim=1)
output_=(-target*torch.log(input_softmax)).sum(dim=1)
print(output_)
print(torch.allclose(output_,output))
```

```python
import torch.nn as nn
import torch
import torch.nn.functional as F

#target比input少一维度,3D情况
loss = nn.CrossEntropyLoss(reduction='none')
input = torch.randn(3,5,6)
target = torch.empty((3,6), dtype=torch.long).random_(input.shape[1])
output = loss(input, target)
print(output)

target=F.one_hot(target,input.shape[1]).permute(0,2,1)

input_softmax=F.softmax(input,dim=1)
output_=(-target*torch.log(input_softmax)).sum(dim=1)
print(output_)
print(torch.allclose(output_,output))
```

```python
import torch.nn as nn
import torch
import torch.nn.functional as F

#target比input少一维度,4D情况
loss = nn.CrossEntropyLoss(reduction='none')
input = torch.randn(3,5,6,7)
target = torch.empty((3,6,7), dtype=torch.long).random_(input.shape[1])
output = loss(input, target)
print(output)

target=F.one_hot(target,input.shape[1]).permute(0,3,1,2)

input_softmax=F.softmax(input,dim=1)
output_=(-target*torch.log(input_softmax)).sum(dim=1)
print(output_)
print(torch.allclose(output_,output))
```



# nn.CTCLoss

# nn.NLLLoss

# nn.PoissonNLLLoss

# nn.GaussianNLLLoss

# nn.KLDivLoss

# nn.BCELoss

- **Binary Cross Entropy二元交叉熵**

  要点：

  1. 输入数据`input`和`target`的每个元素互相对应，**一个元素算做一个样本**：计算两种可能，一是元素的取值$input[i]$和$target[i]$，二是元素取反$(1-input[i])$和$(1-target[i])$，只需要提供数据，接口API也会自动计算对应元素取反计算交叉熵
  2. 输入数据`input`和`target`形状必须相同
  3. 前面的取反操作要求：输入数据`input`和`target`的元素取值范围在[0,1]之间
  4. `nn.BCELoss`和`nn.BCEWithLogitsLoss`唯一不同的地方在于，`nn.BCEWithLogitsLoss`会在内部对`input`进行`sigmoid`操作（元素自己）。要求输入数据元素取值范围在[0,1]之间是固有要求。内部自带的`sigmoid`函数可以让`input`满足这一点。换言之：`nn.BCELoss`要求输入数据`input`和`target`的元素取值范围在[0,1]之间。`nn.BCEWithLogitsLoss`只要求`target`的元素取值范围在[0,1]之间，`input`没有这样的要求，因为内部的`sigmoid`可以完成。

```python
torch.nn.BCELoss(weight=None, 
                 size_average=None, 
                 reduce=None, 
                 reduction='mean'
                )(input,target)
```

还原计算过程

```python
import torch.nn as nn
import torch

#API计算
loss = nn.BCELoss()
# 区间[0, 1)内均匀分布的随机数
input = torch.rand(3, 2)
target = torch.rand(3, 2)
output = loss(input, target)

#还原
input_reverse=1-input
target_reverse=1-target
output_=(-target*torch.log(input)-target_reverse*torch.log(input_reverse)).mean()
print(output_==output) #True
```



- 参数

  

# nn.BCEWithLogitsLoss

- **Binary Cross Entropy with logits 带logits的二元交叉熵**

  要点：

  1. 输入数据`input`和`target`的每个元素互相对应，**一个元素算做一个样本**：计算两种可能，一是元素的取值$input[i]$和$target[i]$，二是元素取反$(1-input[i])$和$(1-target[i])$，只需要提供数据，接口API也会自动计算对应元素取反计算交叉熵
  2. 输入数据`input`和`target`形状必须相同
  3. 前面的取反操作要求：输入数据`input`和`target`的元素取值范围在[0,1]之间
  4. `nn.BCELoss`和`nn.BCEWithLogitsLoss`唯一不同的地方在于，`nn.BCEWithLogitsLoss`会在内部对`input`进行`sigmoid`操作（元素自己）。要求输入数据元素取值范围在[0,1]之间是固有要求。内部自带的`sigmoid`函数可以让`input`满足这一点。换言之：`nn.BCELoss`要求输入数据`input`和`target`的元素取值范围在[0,1]之间。`nn.BCEWithLogitsLoss`只要求`target`的元素取值范围在[0,1]之间，`input`没有这样的要求，因为内部的`sigmoid`可以完成。
  5. **"logits" 是指模型的原始输出，通常是在未经过激活函数（如 softmax 或 sigmoid）处理的模型输出值。所以严谨来说，这个命名不对。**
  6.  对input进行**Sigmoid +BCELoss** 

```python
torch.nn.BCEWithLogitsLoss(weight=None, 
                           size_average=None, 
                           reduce=None, 
                           reduction='mean', 
                           pos_weight=None
                          )(input,target)
```

还原计算过程

```python
import torch.nn.functional as F

loss = nn.BCEWithLogitsLoss()
#标准正太分布，取值不一定是[0,1]
input = torch.randn(3, 2)
target = torch.randn(3, 2)
output = loss(input, target)

input=torch.sigmoid(input)

input_reverse=1-input
target_reverse=1-target
output_=(-target*torch.log(input)-target_reverse*torch.log(input_reverse)).mean()
print(output_==output)
```



# nn.MarginRankingLoss

# nn.HingeEmbeddingLoss

# nn.MultiLabelMarginLoss

# nn.HuberLoss

# nn.SmoothL1Loss

# nn.SoftMarginLoss

# nn.MultiLabelSoftMarginLoss

# nn.CosineEmbeddingLoss

# nn.MultiMarginLoss

# nn.TripleMarginLoss

# nn.TripleMarginWithDistanceLoss

