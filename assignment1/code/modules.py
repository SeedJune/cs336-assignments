import torch
import einops
import torch.nn as nn
import math
from typing import Optional, BinaryIO, IO
from collections.abc import Callable, Iterable
import numpy as np
import numpy.typing as npt
import os

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, device=device, dtype=dtype))
        self.sigma = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=self.sigma, a=-3 * self.sigma, b = 3 * self.sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model,device=device, dtype=dtype))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        denominator = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        output = x / denominator * self.weight
        return output.to(in_dtype)

class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(d_model / 3 * 8)
            mol = self.d_ff % 64
            if mol < 32:
                self.d_ff -= mol
            else:
                self.d_ff += (64 - mol)
        else:
            self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        silu_x = x1 * torch.sigmoid(x1)
        x2 = torch.mul(silu_x, x3)
        output = self.w2(x2)
        return output

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        indexes = torch.arange(0, max_seq_len, dtype=torch.float32, device=device)
        thetas = torch.pow(theta, (2 * torch.arange(1, d_k/2 + 1, dtype=torch.float32, device=device) - 2) / d_k)
        parameters = torch.unsqueeze(indexes, dim=1) / torch.unsqueeze(thetas, dim=0)
        sin_theta = torch.sin(parameters)
        cos_theta = torch.cos(parameters)
        self.register_buffer("sin_theta", sin_theta, persistent=False)
        self.register_buffer("cos_theta", cos_theta, persistent=False)


    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        sin = self.sin_theta[token_positions]
        cos = self.cos_theta[token_positions]
        x_odd = x[..., 0::2]
        x_even = x[..., 1::2]
        rotated_odd = x_odd * cos - x_even * sin
        rotated_even = x_odd * sin + x_even * cos
        output = torch.stack([rotated_odd, rotated_even], dim=-1).flatten(-2)
        return output.to(dtype=x.dtype)


class MHA(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None):
        super().__init__()
        self.d = d_model
        self.h = num_heads
        self.dk = int(d_model / num_heads)
        self.output_proj = Linear(self.d, self.d)
        self.q_proj = Linear(self.d, self.d)
        self.k_proj = Linear(self.d, self.d)
        self.v_proj = Linear(self.d, self.d)
        if theta is not None:
            self.rope = RoPE(theta, self.dk, max_seq_len)
        else:
            self.rope = None

    def forward(self, x:torch.Tensor, token_positions=None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = einops.rearrange(Q, "b s (h d) -> b h s d", h=self.h)
        K = einops.rearrange(K, "b s (h d) -> b h s d", h=self.h)
        V = einops.rearrange(V, "b s (h d) -> b h s d", h=self.h)
        seq_len = Q.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        mask = ~torch.triu(mask, diagonal=1)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long) 
            token_positions = token_positions.unsqueeze(-2)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        attn_score = scaled_dot_product_attention(Q, K, V, mask=mask)
        output = einops.rearrange(attn_score, "b h s d -> b s (h d)")
        output = self.output_proj(output)
        return output

class Transformer_Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=1e-5)
        self.ln2 = RMSNorm(d_model, eps=1e-5)
        self.attn = MHA(d_model, num_heads, theta, max_seq_len)
        self.ffn = FFN_SwiGLU(d_model, d_ff)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        output = x + self.ffn(self.ln2(x))
        return output

class Transformer_LM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Transformer_Block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, eps=1e-5)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for func in self.layers:
            x = func(x)
        x = self.ln_final(x)
        output = self.lm_head(x)
        return output



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invaild learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0) + 1
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                alpha_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data = p.data - lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                p.data -= alpha_t * (m / (torch.sqrt(v) + eps))
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss


def Softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    maximum = torch.max(x, dim=dim, keepdim=True).values
    numerator = torch.exp(x - maximum)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator

def scaled_dot_product_attention(Q, K, V, mask=None):
    tmp = (einops.einsum(Q, K, "... n d, ... m d -> ... n m"))/math.sqrt(Q.shape[-1])
    if mask is not None:
        mask = ~mask
        tmp[..., mask] = float('-inf')
    x = Softmax(tmp, dim=-1)
    output = einops.einsum(x, V, "... n m, ... m d -> ... n d")
    return output

def cross_entropy(logits:torch.Tensor, targets:torch.Tensor):
    maximum = torch.max(logits, dim=-1, keepdim=True).values
    logits = logits - maximum
    logsum = torch.log(torch.sum(torch.exp(logits), dim=-1))
    target_logits = torch.gather(logits, index=targets.unsqueeze(-1), dim=-1).squeeze(-1)
    output = logsum - target_logits
    return output.mean()

def Cosine_Annealing_Schedule(t, alpha_max, alpha_min, t_w, t_c):
    if t < t_w:
        return alpha_max * (t / t_w)
    if t > t_c:
        return alpha_min
    w = (t - t_w) / (t_c - t_w)
    return alpha_min + 0.5 * (1 + math.cos(w * math.pi)) * (alpha_max - alpha_min)

@torch.no_grad()
def Gradient_Clipping(parameters, M, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 
    norm = torch.stack([
        torch.linalg.vector_norm(grad)
        for grad in grads
    ])
    conbined_norm = torch.linalg.vector_norm(norm)
    factor = (M / (conbined_norm + eps)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(factor)

def Data_Loading(x:npt.NDArray, batch_size:int, context_length:int, device:str) -> tuple[torch.Tensor, torch.Tensor]:
    m = len(x)
    start = np.random.randint(low=0, high=(m - context_length), size=(batch_size, 1))
    index = start + np.arange(context_length)
    inputs = torch.tensor(x[index], dtype=torch.long, device=device)
    targets = torch.tensor(x[index + 1], dtype=torch.long, device=device)
    return (inputs, targets)

def save_checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out:str | os.PathLike | BinaryIO | IO[bytes]):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(state_dict, out)
    return 

def load_checkpoint(src:str | os.PathLike | BinaryIO | IO[bytes], model:nn.Module, optimizer:torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict=state_dict["model"])
    optimizer.load_state_dict(state_dict=state_dict["optimizer"])
    return state_dict["iteration"]
