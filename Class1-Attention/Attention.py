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

vocab_size = 1000  # 词汇表大小
embed_dim = 512     # 嵌入维度
batch_size = 32     # 批量大小
seq_len = 50        # 序列长度

embedding_layer = EmbeddingLayer(vocab_size, embed_dim)
attention_layer = Attention(embed_dim)

input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]

input_repr = embedding_layer(input_seq)  # [batch_size, seq_len, embed_dim]

output, attention_weights = attention_layer(input_repr)

print("Input Representation Shape:", input_repr.shape)
print("Output Shape:", output.shape)
print("Attention Weights Shape:", attention_weights.shape)