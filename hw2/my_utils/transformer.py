import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


vocab_size = 128 + 1 # <bos> + vqvae's 128 vocabulary
index_bos = vocab_size - 1
qH, qW = 8, 8 # the height and width of the quantized image
L = qH * qW + 1 
dmodel = 128 # the dimension of the embedding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def append_bos(x):
    '''append <bos> token to the input x'''
    B, L = x.shape
    bos = torch.full((B, 1), index_bos, dtype=torch.long)
    return torch.cat([bos, x], dim=1)
    
class TSFM_DataLoader(DataLoader):
    '''
    Load the data then return the quantized data through vqvae
    args:
        data: (n, H, W, C) uint8
        quantizer: the quantizer model(vqvae)
        batch_size: the batch size
        shuffle: whether to shuffle the data
        num_workers: the number of workers
    '''
    def __init__(self, data, quantizer, batch_size, shuffle=True, num_workers=0, index_bos=index_bos, ):
        '''
        data: (n, H, W, C) uint8
        '''
        super().__init__(data, batch_size, shuffle=shuffle, num_workers=num_workers)
        self.index_bos = index_bos
        self.quantizer = quantizer

    @torch.no_grad() 
    def __iter__(self) :
        '''
        return x: (B, L) torch.long
        '''
        for x in super().__iter__():
            B, _, _, _ = x.shape
            x = x.to(device)
            xarray = self.quantizer.quantize(x) # (B, qH, qW)
            xarray = xarray.reshape(B, qH * qW)   
            x = torch.tensor(xarray, dtype=torch.long)            
            x = append_bos(x)
            yield x # (B, L)
                        
class Attention(nn.Module):
    def __init__(self, L, head_size):
        '''head_size: the dimension of the projected embedding for each head.
        '''
        super().__init__()
        self.q = nn.Linear(dmodel, head_size, bias=False)
        self.k = nn.Linear(dmodel, head_size, bias=False)
        self.v = nn.Linear(dmodel, head_size, bias=False)
        self.head_size= head_size
        self.scale = head_size **(-0.5)
        self.register_buffer('tril', torch.tril(torch.ones(L, L))) # a lower triangular matrix for masking

    def forward(self, x):
        '''x: (B, L, dim)'''
        B, L, _ = x.shape
        q = self.q(x) # (B, L, dim)
        k = self.k(x) 
        v = self.v(x)
        wei = q @ k.transpose(-2, -1) * self.scale # (B, L, L) The softmax(QK^T/sqrt(d)) in the paper
        wei = wei.masked_fill(self.tril == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)  
        y = wei @ v
        return y
    

class MultiHeadAttention(nn.Module):
    def __init__(self, L, head_size, n_head):
        super().__init__()
        self.heads = nn.ModuleList([Attention(L,head_size) for _ in range(n_head)])
    
    def forward(self, x):
        '''x: (B, L, head_size*n_head)'''
        y = torch.cat([h(x) for h in self.heads], dim=-1) 
        return y # (B, L, head_size*n_head)
  
class TransformerBlock(nn.Module):
    def __init__(self, L, head_size, n_head):
        super().__init__()
        self.attention= MultiHeadAttention(L, head_size, n_head)
        self.norm1, self.norm2 = nn.LayerNorm(dmodel), nn.LayerNorm(dmodel),
        self.mlp = nn.Sequential(nn.Linear(dmodel, dmodel), nn.GELU())

    def forward(self, x):
        '''x: (B, L, dmodel)'''
        x = self.attention(self.norm1(x)) + x
        y = self.mlp(self.norm2(x)) + x
        return y

class Embedding(nn.Module):
    def __init__(self, vocab_size, L, dmodel):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dmodel)
        self.pos_embed = nn.Embedding(L, dmodel)
    
    def forward(self, x):
        '''x: (B, L),  value in (0, vocab_size)'''
        B, L = x.shape   
        pos = torch.arange(L).to(device) 
        pos = self.pos_embed(pos).unsqueeze(0) # (1, L, dmodel)
        y = self.embed(x)   # (B, L, dmodel)
        y += pos
        return y # (B, L, dmodel)
    
class iGPT(nn.Module):
    def __init__(self, L, vocab_size, dmodel, n_head, n_layer):
        super().__init__()
        self.L, self.vocab_size = L, vocab_size
        self.embedding = Embedding(vocab_size, L, dmodel)
        assert dmodel % n_head == 0, "dmodel must be divisible by n_head"
        self.transformer = nn.Sequential(*[TransformerBlock(L, dmodel // n_head ,n_head) for _ in range(n_layer)])
        self.fc = nn.Linear(dmodel, vocab_size)
    
    def forward(self, x):
        '''
        x: (B, L) 
        return logits: (B, L, vocab_size)
        '''
        x = self.embedding(x) # (B, L, dmodel)
        x = self.transformer(x) # (B, L, dmodel)
        logits = self.fc(x) # (B, L, vocab_size)
        return logits

    def loss(self, x):
        B, L = x.shape
        logits = self(x)[:, :L-1,:] # (B, L-1, vocab_size) Only compare the first L-1 predicted logits
        logits = logits.reshape(B*(L-1), -1) # (B*(L-1), vocab_size)
        target = x[:, 1:].flatten() # (B*(L-1),)
        return F.cross_entropy(logits, target)
    
    @torch.no_grad()
    def sample(self, num_samples):
        '''
        return (num_samples, H * W) int tensor of vocab_size values, <bos> token won't be generated in the samples.
        '''
        # generate a batch of sequences with <bos> at the begining and the rest are zeros
        samples = torch.zeros((num_samples, L)) # (num_samples, L)
        samples[:, 0] = index_bos
        samples = samples.long().to(device)
        for i in range(self.L-1):
            logits = self.forward(samples)
            logits = logits[:, i, :-1]   # (num_samples, vocab_size), the logits for the next token. Remove the last token <bos>
            probs = F.softmax(logits, dim=-1)   
            samples[:, i+1] = torch.multinomial(probs, 1).squeeze()   
        return samples[:, 1:] # (num_samples, L-1)



def main():
    # test the transformer model
    B, L = 2, 65
    x = torch.randint(0, vocab_size-1, (B, L)).to(device)
    model = iGPT(L, vocab_size, dmodel, n_head=4, n_layer=2).to(device)
    print(model(x).shape)
    print(model.loss(x))
    print(model.sample(2))
    print('test passed')

if __name__ == '__main__':
    main()