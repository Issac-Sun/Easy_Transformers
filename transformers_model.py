# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/

#定义位置编码类positional encoding
import math

import torch
from torch import nn

#debug1：   self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i*dimension_of_model)))     #补充除号
#debug2: self.gamma=nn.Parameter(torch.ones(size=(dimension_of_model,)))    #gamma表示权重
        #self.beta=nn.Parameter(torch.zeros(size=(dimension_of_model,)))    #beta表示偏置   torch.zeros(size=)
        #需要一个整数的元组


#编码：位置编码+词根编码
#from transformer.models.embedding.transformer_embedding import TransformerEmbedding
class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionalEncoding(nn.Module):
    def __init__(self,dimension_of_model,max_sequence_len,device):
        super(PositionalEncoding, self).__init__()
        #d_m表示模型的维度     max_len表示序列的最大长度
        self.encoding = torch.zeros(max_sequence_len,dimension_of_model,device=device)
        self.encoding.requires_grad=False
    # >> > torch.zeros(2, 3)
    # tensor([[0., 0., 0.],
    #         [0., 0., 0.]])
    # zeros操作实创建了一个形状为（max_len,d_m）的张量，并且里面所有元素都初始化为0

        pos=torch.arange(start=0,end=max_sequence_len,device=device)#生成序列从0到最大序列长度
        pos=pos.float().unsqueeze(dim=1)#创建第第二维以存放位置信息
        _2i=torch.arange(start=0,end=dimension_of_model,step=2,device=device).float()

        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i*dimension_of_model)))    #针对偶数的位置编码
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i*dimension_of_model)))     #针对奇数

    def forward(self,x):
        #max_len=512 d_model=512
        batch_size,seq_len=x.size() #x是一个二维张量
        return self.encoding[:seq_len,:]    #取出前seq行

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)



#定义多头注意力类 multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self,dimension_of_model,num_head):
        super(MultiHeadAttention, self).__init__()
        self.num_head=num_head
        self.attention=ScaleDotProductAttention()   #TODO：SDPA Done
        self.w_q=nn.Linear(in_features=dimension_of_model,out_features=dimension_of_model)
        self.w_k=nn.Linear(in_features=dimension_of_model,out_features=dimension_of_model)
        self.w_v=nn.Linear(in_features=dimension_of_model,out_features=dimension_of_model)
        self.w_concat=nn.Linear(in_features=dimension_of_model,out_features=dimension_of_model)

    def split(self,tensor):
        batch_size,length,dimension_of_model=tensor.size()
        d_tensor=dimension_of_model//self.num_head  #//表示整除结果为整数
        tensor=tensor.view(batch_size,length,self.num_head,d_tensor).transpose(1,2) #调转头数和len，为了后续的切片方便，每个head独立处理序列
        return tensor

    def concat(self,tensor):
        batch_size,head,length,d_tensor=tensor.size()
        dimension_of_model=d_tensor*head
        tensor=tensor.transpose(1,2).contiguous().view(batch_size,length,dimension_of_model)    #contiguous
        return tensor


    def forward(self,q,k,v,mask=None):
        #对各自的权重做点积
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
        #qkv分头
        q,k,v=self.split(q),self.split(k),self.split(v)
        #算qkv的相似度（similarity）
        output,attention_score=self.attention(q,k,v,mask=mask)  #TODO:attention √
        #拼接，传递至线性层
        output=self.concat(output)      #TODO:concat √
        output=self.w_concat(output)
        return output       #attention_score

#定义缩放内积注意力类 scale dot product attention
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,q,k,v,mask=None,e=1e-12):

        batch_size,head,length,d_tensor=k.size()
        k_transpose=k.transpose(2,3)
        score=(q@k_transpose)/math.sqrt(d_tensor)
        #是否使用mask
        if mask is not None:
            score=score.masked_fill_(mask=0,value=-10000)
        #masked_fill 是PyTorch中的一个函数，它根据掩码张量中的值将输入张量中的元素替换为指定的值。
        score=self.softmax(score)
        v=v@score
        return v,score

#定义layer norm类
class LayerNorm(nn.Module):
    def __init__(self,dimension_of_model,eps=1e-12):    #eps处理浮点数误差
        super(LayerNorm, self).__init__()
        self.gamma=nn.Parameter(torch.ones(size=(dimension_of_model,)))    #gamma表示权重
        self.beta=nn.Parameter(torch.zeros(size=(dimension_of_model,)))    #beta表示偏置
        self.eps=eps

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)       #针对最后一维，保证输出维度不变
        var=x.var(dim=-1,unbiased=False,keepdim=True)
        output=(x-mean)/math.sqrt(var+self.eps)
        output=self.gamma*output+self.beta      # *是逐元素乘法 @是矩阵乘法
        return output

#定义postionwise feed forward类
class PositionWiseFeedForward(nn.Module):
    def __init__(self,dimension_of_model,hidden,drop_prob=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1=nn.Linear(in_features=dimension_of_model,out_features=hidden)
        self.linear2=nn.Linear(in_features=hidden,out_features=dimension_of_model)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=drop_prob)

    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x

#定义encoder/encoder layer类
class EncoderLayer(nn.Module):
    def __init__(self,dimension_of_model,ffn_hidden,num_head,drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention=MultiHeadAttention(dimension_of_model=dimension_of_model,num_head=num_head)
        self.norm1=LayerNorm(dimension_of_model=dimension_of_model)
        self.dropout1=nn.Dropout(p=drop_prob)
        self.ffn=PositionWiseFeedForward(dimension_of_model=dimension_of_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm2=LayerNorm(dimension_of_model=dimension_of_model)
        self.dropout2=nn.Dropout(p=drop_prob)

    def forward(self,x,src_mask):   #用于encoder中当句子长度不一时，需要将所有的句子填充至相同的长度。因此在求Q和K的相关性时，
                            # 由于Q和K在encoder中相等，所以src_mask最后表现为右边和下边遮挡（填充1）的矩阵；
        _x=x
        x=self.attention(q=x,k=x,v=x,mask=src_mask) #计算自注意力
        x=self.dropout1(x)          #先正则防止过拟合再归一化减少开销
        x=self.norm1(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        return x

class Encoder(nn.Module):       #一个encoder是由许多encoderlayer组成的
    def __init__(self,encoder_voc_size,max_sequence_len,d_model,ffn_hidden,num_head,num_layer,dropout,device):
        super(Encoder, self).__init__()
        self.embedding=TransformerEmbedding(vocab_size=encoder_voc_size,d_model=d_model,max_len=max_sequence_len
                                            ,drop_prob=dropout,device=device)
        self.layers=nn.ModuleList(modules=[EncoderLayer(dimension_of_model=d_model,ffn_hidden=ffn_hidden,
                                                        num_head=num_head,drop_prob=dropout)
                                  for _ in range(num_layer)])
        #nn.ModuleList表示同时管理多个模块，最后需要for _ in range()

    def forward(self,x,src_mask):
        x=self.emb(x)
        for layer in self.layers:
            x=layer(x,src_mask)
        return x

#定义decoder/decoder layer类
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_head,drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention=MultiHeadAttention(dimension_of_model=d_model,num_head=num_head)
        self.layernorm1=LayerNorm(dimension_of_model=d_model,eps=1e-12)
        self.dropout1=nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention=MultiHeadAttention(dimension_of_model=d_model,num_head=num_head)
        self.layernorm2=LayerNorm(dimension_of_model=d_model,eps=1e-12)
        self.dropout2=nn.Dropout(p=drop_prob)
        self.ffn=PositionWiseFeedForward(dimension_of_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.layernorm3=LayerNorm(dimension_of_model=d_model,eps=1e-12)
        self.dropout3=nn.Dropout(p=drop_prob)

    def forward(self,decoder,encoder,trg_mask,src_mask):        #target_mask source_mask
        _x=decoder
        x=self.self_attention(q=decoder,k=decoder,v=decoder,mask=trg_mask)
        x=self.dropout1(x)
        x=self.layernorm1(x+_x)
        if encoder is not None:
            #计算encoder-decoder attention
            _x=x
            x=self.encoder_decoder_attention(q=x,k=encoder,v=decoder,mask=src_mask)
            x=self.dropout2(x)
            x=self.layernorm2(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.dropout3(x)
        x=self.layernorm3(x+_x)
        return x

class Decoder(nn.Module):
    def __init__(self,decoder_voc_size,max_sequence_len,d_model,ffn_hidden,num_head,num_layer,drop_prob,device):
        super(Decoder, self).__init__()
        self.embedding=TransformerEmbedding(vocab_size=decoder_voc_size,d_model=d_model,max_len=max_sequence_len,
                                            drop_prob=drop_prob,device=device)
        self.layers=nn.ModuleList([DecoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,num_head=num_head,
                                                drop_prob=drop_prob)
                                   for _ in range(num_layer)])
        self.linear=nn.Linear(in_features=d_model,out_features=decoder_voc_size)

    def forward(self,target,source,trg_mask,src_mask):
        target=self.embedding(target)
        for layer in self.layers:
            target=layer(target,source,trg_mask,src_mask)   #在每次迭代中，将目标序列的嵌入向量和源序列的嵌入向量以及两个掩码传递给当前层，
                                                            # 并将输出赋值回trg
        output=self.linear(target)
        return output

class TransformerModel(nn.Module):
    def __init__(self, encoder_voc_size, decoder_voc_size, max_sequence_len, d_model, ffn_hidden, num_head, num_layers, dropout, device):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(encoder_voc_size, max_sequence_len, d_model, ffn_hidden, num_head, num_layers, dropout, device)
        self.decoder = Decoder(decoder_voc_size, max_sequence_len, d_model, ffn_hidden, num_head, num_layers, dropout, device)

    def forward(self, src, trg, src_mask, trg_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, encoder_output, trg_mask, src_mask)
        return decoder_output

# 实例化模型
encoder_voc_size = 10000  # 例如，编码器词汇表大小为10000
decoder_voc_size = 10000  # 解码器词汇表大小也为10000
max_sequence_len = 100    # 序列的最大长度为100
d_model = 512             # 模型的维度为512
ffn_hidden = 2048         # 前馈神经网络的隐藏层维度为2048
num_head = 8              # 多头自注意力机制中的头数为8
num_layers = 6            # 编码器和解码器的层数为6
dropout = 0.1             # dropout概率为0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(encoder_voc_size, decoder_voc_size, max_sequence_len, d_model, ffn_hidden, num_head, num_layers, dropout, device)

# 打印模型结构
print(model)














