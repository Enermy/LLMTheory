# Attention-
介绍一下注意力机制的原理与QKV的计算，并附上代码实现的详细过程

深度学习中的注意力机制是指人工神经网络中的一种机制，它能够帮助模型更好地关注重要的信息，而忽略不重要的信息。

​	这种机制的作用类似于生物体中的注意力机制，可以帮助我们过滤出重要的信息，并将这些信息传递给相应的神经元进行处理。

​	深度学习中的注意力机制与生物学中的注意力机制的关系是相似的，但并不完全相同。在深度学习中，注意力机制主要是通过自然语言处理、视觉和听觉处理等应用来实现的。它通过学习模型来自动判断哪些信息更重要，并将信息传递给相应的神经元进行处理。生物体中的注意力机制是由两个神经系统——前额叶皮层和脑干辅助系统——协同工作完成的，而深度学习中的注意力机制则是通过人工神经网络来实现的。

​	在自然语言处理中，注意力机制可以帮助模型更好地理解句子的意思，并提取出重要的信息。例如，在翻译任务中，注意力机制可以帮助模型更好地理解源语言的句子，并将其翻译成目标语言。在问答任务中，注意力机制可以帮助模型更好地理解问题的意思，并找到相应的答案。在视觉和听觉处理中，注意力机制也有广泛的应用。

### 1、自注意力机制
自注意力机制就允许把 it 和 animal 联系到一起，具体来说就是通过查看输入序列中其他位置以便寻找更好的编码 it 的线索。

​	不同颜色深度就表示权重的大小，实际上相当于把输入序列进行了一个预处理的过程。
 <img src="https://github.com/user-attachments/assets/b3fef600-ab05-450d-8bec-ef659135152f" width="500" />

自注意力机制有很多优势：

- 可以让模型聚焦在输入的重要部分而忽略其他不相关信息；
- 它能够处理变长的序列数据（文本、图像等），而且可以在不改变模型结构的情况下加强模型的表示能力，还能减少模型的计算复杂度，因为它只对关键信息进行处理。

  #### 自注意力机制的计算
  ##### 获取Q K V值
  从每个编码器的输入向量，创建 Q K V 向量。具体来说就是把 词嵌入 Embedding 向量乘以训练得到的三个矩阵 W。Query可以理解成解码器中前一时刻的状态，Key可以理解成编码器的隐状态，二者之间可以求向量的相似度，也就是注意力的分数。

   <img src="https://github.com/user-attachments/assets/d20f7de9-48c0-4061-9df3-173a4264af7e" width="800" />
##### 自注意力分数
假设我们正在计算Thinking的自注意力，需要根据这个词对输入句子的每一个词进行评分，相当于是看 Thinking 和其他每一个词的关联度。通过 query 和 各个单词的 key 进行点积运算。 其实和之前的键值对注意力机制可以类比，只不过现在的query和输入其实是一个序列。
<img src="https://github.com/user-attachments/assets/ed876552-0022-49be-91d2-537937b241bf" width="600" />

##### Softmax归一化
除以key向量维度的平方根，目的是为了使训练中梯度更加稳定。这个softmax分数就叫做注意力分布，他表示Thinking对每个位置的关注程度。
<img src="https://github.com/user-attachments/assets/32c140d7-dd65-4eec-9b06-233072488c5d" width="600" />

##### 注意力加权求和
将每个value向量 乘以 softmax分数。然后通过加权求和得到自注意力的输出 z1 ，生成的向量再进一步发送到前馈神经网络，实际实现中计算都是以矩阵形式来完成的，效率更高。
<img src="https://github.com/user-attachments/assets/66b95086-cba2-4969-8a71-f27e8bcbbf72" width="600" />

##### 自注意力的矩阵计算
 使用矩阵运算可以一次性计算出所有位置的Attention输出向量。
<img src="https://github.com/user-attachments/assets/56a3d68b-1c29-4e0f-bf15-476fefcf34e4" width="600" />

##### 自注意力的理解
计算输入序列每个元素与其余元素之间的相关性得分。
<img src="https://github.com/user-attachments/assets/bf7ad8c5-f4d9-45cf-bd21-060627d470da" width="600" />

## 2、注意力机制代码实现
接下来将展示注意力机制代码的基本流程：

##### 步骤1，导入必要的库

在这个步骤中，我们需要导入必要的库，例如 numpy 和 torch。numpy 用于数值计算，torch 用于深度学习。

```Bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```
定义一个简单的嵌入层
```
import numpy as np  # 用于数值计算* 
import torch       # 用于深度学习* 
import torch.nn as nn  # 导入 PyTorch 的神经网络模块* 
import torch.nn.functional as F  # 导入 PyTorch 的函数库*
import math

# 定义一个简单的嵌入层
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)
```
##### 步骤2，创建一个注意力类
在 Python 中，我们可以使用 nn.Module 创建一个注意力类。这个类将包含我们注意力机制的实现。
```
class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # 计算 Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(embed_dim)  # [batch_size, seq_len, seq_len]

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # 加权求和
        output = torch.matmul(attention_weights, V)  # [batch_size, seq_len, embed_dim]

        return output, attention_weights
```
设置一些参数：
```
vocab_size = 1000  # 词汇表大小
embed_dim = 512     # 嵌入维度
batch_size = 32     # 批量大小
seq_len = 50        # 序列长度
```
初始化嵌入层和注意力层
```
embedding_layer = EmbeddingLayer(vocab_size, embed_dim)
attention_layer = Attention(embed_dim)
```
然后随机生成一个输入序列，将该序列进行embedding编码后，运用注意力层进行计算：

```python
input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]

input_repr = embedding_layer(input_seq)  # [batch_size, seq_len, embed_dim]

output, attention_weights = attention_layer(input_repr)
```

最后打印输出结果：

```python
print("Input Representation Shape:", input_repr.shape)
print("Output Shape:", output.shape)
print("Attention Weights Shape:", attention_weights.shape)
```

输出如下：

```
Input Representation Shape: torch.Size([32, 50, 512])
Output Shape: torch.Size([32, 50, 512])
Attention Weights Shape: torch.Size([32, 50, 50])
```
