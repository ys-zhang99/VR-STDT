import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel
from src.layers.embedding import AirEmbedding
import numpy as np

dartboard_map = {0: '50-200',
                 1: '50-200-500',
                 2: '50',
                 3: '25-100-250'}

class LatentLayer(nn.Module):  
    '''
    The latent layer to compute mean and std
    '''
    def __init__(self,
                 dm_dim,  # the dimension of deterministic states
                 latent_dim_in,  # the dimension of input latent variables
                 latent_dim_out,  # the dimension of output latent variables
                 hidden_dim,  # the intermediate dimension
                 num_layers=2):
        super(LatentLayer, self).__init__()

        self.num_layers = num_layers
        self.enc_in = nn.Sequential(
            nn.Conv2d(dm_dim+latent_dim_in, hidden_dim, 1))

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU(inplace=True))
        self.enc_hidden = nn.Sequential(*layers)
        self.enc_out_1 = nn.Conv2d(hidden_dim, latent_dim_out, 1)
        self.enc_out_2 = nn.Conv2d(hidden_dim, latent_dim_out, 1)

    def forward(self, x):
        # x: [b, c, n, t]
        h = self.enc_in(x)
        for i in range(self.num_layers):
            h = self.enc_hidden[i](h)
        mu = torch.minimum(self.enc_out_1(h), torch.ones_like(h)*10)
        sigma = torch.minimum(self.enc_out_2(h), torch.ones_like(h)*10)
        return mu, sigma


class StochasticModel(nn.Module):
    '''
    The generative model.
    The inference model can also use this implementation, while the input should be shifted
    '''
    def __init__(self,
                 dm_dim,  # the dimension of the deterministic states
                 latent_dim,  # the dimension of the latent variables
                 num_blocks=4):

        super(StochasticModel, self).__init__()
        self.layers = nn.ModuleList()

        # the bottom n-1 layers
        for _ in range(num_blocks-1):
            self.layers.append(
                LatentLayer(dm_dim,
                            latent_dim,
                            latent_dim,
                            latent_dim,
                            2))
        # the top layer
        self.layers.append(
            LatentLayer(dm_dim,
                        0,
                        latent_dim,
                        latent_dim,
                        2))

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma, requires_grad=False)
        return mu + eps*sigma

    def forward(self, d):
        # d: [num_blocks, b, c, n, t]
        # top-down
        _mu, _logsigma = self.layers[-1](d[-1])
        _sigma = torch.exp(_logsigma) + 1e-3  # for numerical stability
        mus = [_mu]
        sigmas = [_sigma]
        z = [self.reparameterize(_mu, _sigma)]

        for i in reversed(range(len(self.layers)-1)):
            _mu, _logsigma = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            _sigma = torch.exp(_logsigma) + 1e-3
            mus.append(_mu)
            sigmas.append(_sigma)
            z.append(self.reparameterize(_mu, _sigma))

        z = torch.stack(z)
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        return z, mus, sigmas

class AirFormer(BaseModel):
    '''
    the AirFormer model
    '''
    def __init__(self,
                 dropout=0.3,  # dropout rate
                 spatial_flag=True,  # whether to use DS-MSA
                 stochastic_flag=False,  # whether to use latent vairables
                 hidden_channels=32,  # hidden dimension
                 end_channels=512,  # the decoder dimension
                 blocks=6,  # the number of stacked AirFormer blocks
                 mlp_expansion=1,  # the mlp expansion rate in transformers
                 num_heads=2,  # the number of heads
                 dartboard=0,  # the type of dartboard
                 **args):
        super(AirFormer, self).__init__(**args)
        self.dropout = dropout
        self.blocks = blocks
        self.spatial_flag = spatial_flag
        self.stochastic_flag = stochastic_flag
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.embedding_air = AirEmbedding()
        self.alpha = 10  # the coefficient of kl loss

        self.get_dartboard_info(dartboard)

        # a conv for converting the input to the embedding
        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=hidden_channels,
                                    kernel_size=(1, 1))
        #二维卷积做patching,[b,c,n,t]--[b,c,h,w]:在n上步长为1，t方向5步合并
        self.patching = nn.Conv2d(in_channels=self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(1,5),
                                  stride=(1,5))

        for b in range(blocks):
            window_size = self.seq_len // 2 ** ((blocks - b - 1)//2)
            #window_size = self.seq_len
            self.t_modules.append(CT_MSA(hidden_channels,
                                         depth=1,
                                         heads=num_heads,
                                         window_size=window_size,
                                         mlp_dim=hidden_channels*mlp_expansion,
                                         num_time=self.seq_len, device=self.device))

            if self.spatial_flag:
                self.s_modules.append(DS_MSA(hidden_channels,
                                             depth=1,
                                             heads=num_heads,
                                             mlp_dim=hidden_channels*mlp_expansion,
                                             assignment=self.assignment,
                                             num_sectors = self.num_nodes,
                                             mask=self.mask,
                                             dropout=dropout))
            else:
                self.residual_convs.append(nn.Conv1d(in_channels=hidden_channels,
                                                     out_channels=hidden_channels,
                                                     kernel_size=(1, 1)))

            self.bn.append(nn.BatchNorm2d(hidden_channels))

        # create the generrative and inference model
        if stochastic_flag:
            self.generative_model = StochasticModel(
                hidden_channels, hidden_channels, blocks)
            self.inference_model = StochasticModel(
                hidden_channels, hidden_channels, blocks)

            self.reconstruction_model = \
                nn.Sequential(nn.Conv2d(in_channels=hidden_channels*blocks,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(in_channels=end_channels,
                                        out_channels=self.input_dim,
                                        kernel_size=(1, 1),
                                        bias=True)
                              )

        # create the decoder layers
        if self.stochastic_flag:
            self.end_conv_1 = nn.Conv2d(in_channels=hidden_channels*blocks*2,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)
        else:
            self.end_conv_1 = nn.Conv2d(in_channels=hidden_channels*blocks,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=self.horizon*self.output_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.linear1 = nn.Sequential(
            nn.Linear(8,hidden_channels,bias=True),
            nn.ReLU(),
            nn.Linear(hidden_channels,hidden_channels,bias=True),
            nn.ReLU(),
            nn.Linear(hidden_channels,8,bias=True)
        )

    def get_dartboard_info(self, dartboard):
        '''
        get dartboard-related attributes
        '''
        path_assignment = 'data/local_partition/' + \
            dartboard_map[dartboard] + '/assignment.npy'
        path_mask = 'data/local_partition/' + \
            dartboard_map[dartboard] + '/mask.npy'
        print(path_assignment)
        self.assignment = torch.from_numpy(
            np.load(path_assignment)).float().to(self.device)
        self.mask = torch.from_numpy(
            np.load(path_mask)).bool().to(self.device)

    def forward(self, inputs, supports=None):
        '''
        inputs: the historical data
        supports: adjacency matrix (actually our method doesn't use it)
                Including adj here is for consistency with GNN-based methods
        '''
        #x_embed = self.embedding_air(inputs[..., 11:15].long())
        #x = torch.cat((inputs[..., :11], x_embed, inputs[..., 15:]), -1)
        x = inputs.permute(0, 3, 2, 1)  # [batch, channel, nodes, times]
        #x = self.patching(x)
        x = self.start_conv(x)
        d = []  # deterministic states


        for i in range(self.blocks):
            if self.spatial_flag:
                x = self.s_modules[i](x)
            else:
                x = self.residual_convs[i](x)      
            x = self.t_modules[i](x)  # [b, c, n, t]

            x = self.bn[i](x)
            d.append(x)


        d = torch.stack(d)  # [num_blocks, b, c, n, t]

        # generatation and inference
        if self.stochastic_flag:
            d_shift = [(nn.functional.pad(d[i], pad=(1, 0))[..., :-1])
                       for i in range(len(d))]
            d_shift = torch.stack(d_shift)  # [num_blocks, b, c, n, t]
            z_p, mu_p, sigma_p = self.generative_model(
                d_shift)  # run the generative model
            z_q, mu_q, sigma_q = self.inference_model(
                d)  # run the inference model

            # compute kl divergence loss
            p = torch.distributions.Normal(mu_p, sigma_p)
            q = torch.distributions.Normal(mu_q, sigma_q)
            kl_loss = torch.distributions.kl_divergence(
                q, p).mean() * self.alpha

            # reshaping
            num_blocks, B, C, N, T = d.shape
            z_p = z_p.permute(1, 0, 2, 3, 4).reshape(
                B, -1, N, T)  # [B, num_blocks*C, N, T]
            z_q = z_q.permute(1, 0, 2, 3, 4).reshape(
                B, -1, N, T)  # [B, num_blocks*C, N, T]

            # reconstruction
            x_rec = self.reconstruction_model(z_p)  # [b, c, n, t]
            x_rec = x_rec.permute(0, 3, 2, 1)

            # prediction
            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(
                B, -1, N, T)  # [B, num_blocks*C, N, T]
            x_hat = torch.cat([d[..., -1:], z_q[..., -1:]], dim=1)
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
            return x_hat, x_rec, kl_loss

        else:
            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(
                B, -1, N, T)  # [B, num_blocks*C, N, T]
            x_hat = F.relu(d[...,-1:]) #[256,128*6,8,6]
            x_hat = F.relu(self.end_conv_1(x_hat))#[256,1024,8,1]
            x_hat = self.end_conv_2(x_hat)#[256,1,8,1]
            x_hat=x_hat.reshape(B,self.horizon,N,self.output_dim)
            x_hat = torch.permute(x_hat,(0,1,3,2))
            x_hat = self.linear1(x_hat)
            x_hat=torch.permute(x_hat,(0,1,3,2))
            return x_hat

class SpatialAttention(nn.Module):
    def __init__(self,heads,dim,dropout,num_sectors=8):
        super().__init__()

        self.channel = nn.Sequential(
            nn.Linear(dim,dim//heads,bias=True),
            nn.ReLU(),
            nn.Linear(dim//heads,dim,bias=True),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Linear(num_sectors,dim,bias=True),
            nn.ReLU(),
            nn.Linear(dim,num_sectors,bias=True),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        B, N, C = x.shape

        """ avg_channel = torch.mean(x,dim=1)
        avg_channel = torch.reshape(avg_channel,(B,C))
        max_channel,_ = torch.max(x,dim=1)
        max_channel = torch.reshape(max_channel,(B,C))
        avg_channel_out = self.channel(avg_channel)
        max_channel_out = self.channel(max_channel)
        channel_out = torch.reshape(self.sigmoid(avg_channel_out+max_channel_out),(B,1,C))
        x = x*channel_out.expand_as(x) """

        avg_spatial = torch.mean(x,dim=2)
        avg_spatial = torch.reshape(avg_spatial,(B,N))
        max_spatial,_ = torch.max(x,dim=2)
        max_spatial = torch.reshape(max_spatial,(B,N))
        avg_spatial_out = self.spatial(avg_spatial)
        max_spatial_out = self.spatial(max_spatial)
        spatial_out = torch.reshape(self.sigmoid(avg_spatial_out+max_spatial_out),(B,N,1))
        x = x * spatial_out.expand_as(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
""" class SpatialAttention(nn.Module):
    def __init__(self,heads,dim,dropout,num_sectors=8):
        super().__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.channel = nn.Sequential(
            nn.Linear(dim,dim//heads,bias=True),
            nn.ReLU(),
            nn.Linear(dim//heads,dim,bias=True),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Linear(num_sectors*self.heads,dim,bias=True),
            nn.ReLU(),
            nn.Linear(dim,num_sectors*self.heads,bias=True),
            nn.Sigmoid()
        )
        ##channel压缩实际是将spatial压缩到一维##
        self.channel_squeeze = nn.Conv1d(
            in_channels = num_sectors * self.heads,
            out_channels = 1,
            kernel_size = 1
        )
        ##spatial压缩实际是将channel压缩到一维##
        self.spatial_squeeze = nn.Conv1d(
            in_channels = dim,
            out_channels = 1,
            kernel_size = 1
        )
        self.proj_channel = nn.Sequential(
            nn.Linear(in_features=dim,out_features=dim),
        )
        self.proj_spatial = nn.Sequential(
            nn.Conv1d(in_channels=num_sectors,out_channels=num_sectors,kernel_size=1),
        )

    def forward(self,x):
        B, N, C = x.shape
        H = self.heads
        x_stack = x
        for i in range(H-1):
            x_stack = torch.cat([x_stack,x],dim=1)
        
        channel_squeeze = self.channel_squeeze(x_stack)
        channel_squeeze = torch.reshape(channel_squeeze,(B,C))
        channel_out = self.channel(channel_squeeze)
        channel_out = torch.reshape(self.sigmoid(channel_out),(B,1,C))
        x_stack = x_stack*channel_out.expand_as(x_stack)

        spatial_squeeze = self.spatial_squeeze(x_stack.permute(0,2,1))
        spatial_squeeze = torch.reshape(spatial_squeeze,(B,N*H))
        spatial_out = self.spatial(spatial_squeeze)
        spatial_out = torch.reshape(self.sigmoid(spatial_out),(B,N*H,1))
        x_stack = x_stack*spatial_out.expand_as(x_stack)
        x_MulHead = x_stack.contiguous().view(B,H,N,C)
        x_out = torch.zeros(x.shape).to('cuda')
        for i in range(H):
            x_out = x_out + x_MulHead[:,i,:,:]
        
        x_out = self.sigmoid(x_out)
        x = self.proj_channel(x)
        #x = self.proj_spatial(x)
        x = self.dropout(x)
        return x """
            

class DS_MSA(nn.Module):
    # Dartboard Spatial MSA
    def __init__(self,
                 dim,  # hidden dimension
                 num_sectors,
                 depth,  # number of MSA in DS-MSA
                 heads,  # number of heads
                 mlp_dim,  # mlp dimension
                 assignment,  # dartboard assignment matrix
                 mask,  # mask
                 dropout=0.):  # dropout rate
        super().__init__()
        self.heads = 3
        self.sigmoid = nn.Sigmoid()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(dim=dim, heads=heads, dropout=dropout,
                                 num_sectors= num_sectors),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(num_sectors,FeedForward(num_sectors, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)  # [b*t, n, c]
        # x = x + self.pos_embedding  # [b*t, n, c]  we use relative PE instead
        for attn, ff1,ff2 in self.layers:
            x = attn(x) + x
            x = ff1(x) + x
            x = x.permute(0,2,1)
            x = ff2(x) + x
            x = x.permute(0,2,1)
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x
    """ def forward(self,x):
         # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)  # [b*t, n, c]
        # x = x + self.pos_embedding  # [b*t, n, c]  we use relative PE instead
        x_stack = x
        for i in range(self.heads-1):
            x_stack = torch.cat([x_stack,x],dim=0) #[m*b*t,n,c]
        x = x_stack
        for attn, ff1 in self.layers:
            x = attn(x) + x
            x = ff1(x) + x
            
        x = torch.reshape(x,(self.heads,b,t,n,c))
        
        x = x.permute(1,0,4,3,2) #[b,m,c,n,t]
        
        
        return x
 """

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True, device=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size)).to(
            device)  # mask for causality

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)  # create local windows
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]

        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:  # reshape to the original size
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class CT_MSA(nn.Module):
    # Causal Temporal MSA
    def __init__(self,
                 dim,  # hidden dim
                 depth,  # the number of MSA in CT-MSA
                 heads,  # the number of heads
                 window_size,  # the size of local window
                 mlp_dim,  # mlp dimension
                 num_time,  # the number of time slot
                 dropout=0.,  # dropout rate
                 device=None):  # device, e.g., cuda
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim,
                                  heads=heads,
                                  window_size=window_size,
                                  dropout=dropout,
                                  device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)  # [b*n, t, c]
        x = x + self.pos_embedding  # [b*n, t, c]
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x

# Pre Normalization in Transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FFN in Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
