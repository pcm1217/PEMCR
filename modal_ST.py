import numpy as np
import torch
from torch import nn
import math
from torch.nn import functional as F
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
  
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   
    return pad_attn_mask.expand(batch_size, len_q, len_k) 
def get_attn_subsequence_mask(seq):

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self,dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) 
        scores.masked_fill_(attn_mask, -1e9) 
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) 

        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,args,d_model,d_k,n_heads,d_v):
        super(MultiHeadAttention, self).__init__()
        self.args=args
        self.d_model=d_model
        self.dk=d_k
        self.n_heads=n_heads
        self.dv=d_v
        self.W_Q = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.dv * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.dv, self.d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.dv).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dk)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,(self.n_heads) * (self.dv)) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args=args
        self.d_model=d_model
        self.d_ff=d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
    def forward(self, inputs):

        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):

    def __init__(self,args,d_model,d_k,n_heads,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args,d_model,d_k,n_heads,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(args,d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):

        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )
    def forward(self, x, mask=None):
    
        out = self.linear(x)
        if mask is not None:  
            out = out.masked_fill(mask, -100000)  
            weight = F.softmax(out, dim=1)
            return weight
        else:
            weight = F.softmax(out, dim=2) 
        return weight 



def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(15, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(15, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x



class PromptLearner(nn.Module):
    def __init__(self, args, item_num, modality_size, loc_numb):
        super().__init__()
        self.args = args
        emb_num = 2
        emb_num_S = 2
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        self.src_emb = nn.Embedding(item_num, args.hidden_units)
        self.loc_emb = nn.Embedding(loc_numb, args.hidden_units)

        self.query_common = nn.Sequential(
            nn.Conv1d(100, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.time_emb = Time2Vec('sin', out_dim=args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.max_len, args.hidden_units)

        self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units)
        self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units)

        drop_out = 0.25
        self.attention_E = AttentionLayer(2 * args.hidden_units, drop_out)

        embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
        embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))

        self.ctx_E = nn.Parameter(embedding_E)
        self.ctx_S_E = nn.Parameter(embedding_S_E)

        self.modality_linear = nn.Linear(args.multimodal_dim, args.hidden_units)
        self.eps = 0.5

    def forward(self, seq, loc_seq, time_seq, img_emb, text_emb, meta_emb):
        seq_feat = self.src_emb(seq)
        positions = torch.arange(seq.shape[1]).expand(seq.shape[0], seq.shape[1]).to(self.args.device)
        seq_feat += self.pos_emb(positions)
        seq_feat = self.emb_dropout(seq_feat)

        img_emb = self.modality_linear(img_emb)
        text_emb = self.modality_linear(text_emb)
        meta_emb = self.modality_linear(meta_emb)

        loc_feat = self.loc_emb(loc_seq)

        img_common_score = self.query_common(img_emb.permute(0, 2, 1))
        text_common_score = self.query_common(text_emb.permute(0, 2, 1))
        meta_common_score = self.query_common(meta_emb.permute(0, 2, 1))
        
        img_common_score = img_common_score.mean(dim=2).unsqueeze(1)
        text_common_score = text_common_score.mean(dim=2).unsqueeze(1)
        meta_common_score = meta_common_score.mean(dim=2).unsqueeze(1)
    
        att_common = torch.cat([img_common_score, text_common_score, meta_common_score], dim=1)
        weight_common = F.softmax(att_common, dim=1)
        common_emb = torch.sum(weight_common.unsqueeze(2) * torch.stack([img_emb, text_emb, meta_emb], dim=1), dim=1)
        sep_img_emb = img_emb - common_emb
        sep_text_emb = text_emb - common_emb
        sep_meta_emb = meta_emb - common_emb

        image_prefer = self.gate_image_prefer(seq_feat)
        text_prefer = self.gate_text_prefer(seq_feat)
        meta_prefer = self.gate_meta_prefer(seq_feat)
        sep_img_emb = torch.multiply(image_prefer, sep_img_emb)
        sep_text_emb = torch.multiply(text_prefer, sep_text_emb)
        sep_meta_emb = torch.multiply(meta_prefer, sep_meta_emb)

        multi_embeds = (common_emb + sep_img_emb + sep_text_emb + sep_meta_emb) / 4

        all_embeds = seq_feat + multi_embeds + loc_feat

        contrastive_loss = self.InfoNCE(seq_feat, multi_embeds, 0.2)

        combined_emb = all_embeds

        ctx_E = self.ctx_E
        ctx_S_E = self.ctx_S_E
        ctx_E_1 = ctx_E

        if ctx_S_E.dim() == 2:
            ctx_E = ctx_E_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1], -1, -1)
            ctx_S_E = ctx_S_E.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1], -1, -1)

        ctx_prefix_E = self.getPrompts(combined_emb.unsqueeze(2), ctx_E, ctx_S_E)
        item_embedding = combined_emb.unsqueeze(2).expand(-1, -1, ctx_prefix_E.shape[2], -1)
        prompt_item = torch.cat((ctx_prefix_E, item_embedding), dim=3)
        at_wt = self.attention_E(prompt_item)
        prompts_E = torch.matmul(at_wt.permute(0, 1, 3, 2), ctx_prefix_E).squeeze()

        return prompts_E, contrastive_loss

    def getPrompts(self, prefix, ctx, ctx_S):
        prompts = torch.cat([ctx, ctx_S, prefix], dim=2)
        return prompts

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=2), F.normalize(view2, dim=2)
        pos_score = torch.sum(view1 * view2, dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.bmm(view1, view2.transpose(1, 2))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=2)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)


class mmckt(nn.Module):
    def __init__(self,args,item_num, loc_num):
        super(mmckt, self).__init__()
        self.args=args
        self.class_=nn.Linear(args.hidden_units,args.all_size)
       
        self.layers = nn.ModuleList([EncoderLayer(self.args,args.hidden_units,args.d_k,args.n_heads,args.d_v,args.d_ff) for _ in range(args.n_layers)])
        if args.Strategy == 'default' :
            self.prompt=PromptLearner(args,item_num, 3, loc_num)

    def phase_one(self, user, log_seqs, loc_seq, time_seq, img_emb, text_emb, meta_emb):
        
        
        enc_outputs, contrastive_loss = self.prompt(log_seqs, loc_seq, time_seq, img_emb, text_emb, meta_emb)

        enc_self_attn_mask = get_attn_pad_mask(log_seqs, log_seqs) # [batch_size, src_len, src_len]

        enc_attn_mask=get_attn_subsequence_mask(log_seqs).to(self.args.device)

        all_mask=torch.gt((enc_self_attn_mask + enc_attn_mask), 0).to(self.args.device)
        
        enc_self_attns = []
        for layer in self.layers:
            
            enc_outputs, enc_self_attn = layer(enc_outputs, all_mask)
    
            enc_self_attns.append(enc_self_attn)
   
        logits=self.class_(enc_outputs[:,-1,:])
        return logits, contrastive_loss, self.prompt.ctx_E,  self.prompt.ctx_S_E
    

    def forward(self,user,log_seqs, loc_seq, time_seq, img_emb, text_emb, meta_emb):
        logits=self.phase_one(user,log_seqs, loc_seq, time_seq, img_emb, text_emb, meta_emb) 
        return logits
    
    def Freeze_a(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():

            if "ctx_S_E" in name:
                param.requires_grad = True
            if "ctx_E" in name:
                param.requires_grad = True
            if "layers" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        self.prompt.pos_emb.requires_grad = True
        self.prompt.modality_linear.requires_grad = True
        self.prompt.query_common.requires_grad = True
        #self.prompt.time_emb.requires_grad = False
        

    def Freeze_b(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():

            if "ctx_S_E" in name:
                param.requires_grad = True
            if "layers" in name:
                param.requires_grad = False
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = False
        self.prompt.pos_emb.requires_grad = False
        self.prompt.modality_linear.requires_grad = False
        self.prompt.query_common.requires_grad = False
        self.prompt.time_emb.requires_grad = False

    def Freeze_c(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():

            if "ctx_S_E" in name:
                param.requires_grad = True
            if "ctx_E" in name:
                param.requires_grad = True
            if "layers" in name:
                param.requires_grad = False
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = False
        self.prompt.pos_emb.requires_grad = False
        self.prompt.modality_linear.requires_grad = False
        self.prompt.query_common.requires_grad = False
        #self.prompt.time_emb.requires_grad = False
        
    def Freeze_d(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():

            if "ctx_S_E" in name:
                param.requires_grad = True
            if "ctx_E" in name:
                param.requires_grad = True
            if "layers" in name:
                param.requires_grad = False
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = False
        self.prompt.pos_emb.requires_grad = False
        self.prompt.modality_linear.requires_grad = True
        self.prompt.query_common.requires_grad = True
        #self.prompt.time_emb.requires_grad = False

    def Freeze_e(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():

            if "ctx_S_E" in name:
                param.requires_grad = True
            if "layers" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = False
        self.prompt.src_emb.requires_grad = False
        self.prompt.pos_emb.requires_grad = False
        self.prompt.modality_linear.requires_grad = True
        self.prompt.query_common.requires_grad = True
        #self.prompt.time_emb.requires_grad = False
