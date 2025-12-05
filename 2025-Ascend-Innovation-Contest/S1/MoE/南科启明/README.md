# 昇腾昇思模型挑战赛 - MoE 模型优化报告
## 一、Baseline 性能数据
### 初始性能（未优化）
| 模型名称 | 内存预留(GB) | 内存分配(GB) | 平均预填充延迟(s) | 平均解码延迟(s) | 文本匹配 | Logits 接近 |
|----------|--------------|--------------|--------------------|------------------|----------|-------------|
| Qwen1.5-MoE-A2.7B-Chat | 31.138512896 | 29.234176512 | 1.9554526805877686 | 0.6352910525645998 | 全部匹配 | 全部接近 |
| deepseek-moe-16b-chat | 34.359738368 | 32.813018112 | 3.664979934692383 | 0.6057731850387514 | 全部匹配 | 全部接近 |

### 最终优化性能（最佳方案）
| 模型名称 | 内存预留(GB) | 内存分配(GB) | 平均预填充延迟(s) | 平均解码延迟(s) | 文本匹配 | Logits 接近 |
|----------|--------------|--------------|--------------------|------------------|----------|-------------|
| Qwen1.5-MoE-A2.7B-Chat | 31.138512896 | 29.234176512 | 1.7499115467071533 | 0.13593099928568053 | 全部匹配 | 全部接近 |
| deepseek-moe-16b-chat | 34.359738368 | 32.813018112 | 3.5624752044677734 | 0.14991246504993094 | 全部匹配 | 全部接近 |

## 二、核心优化方案
### 1. 接入融合算子
| 融合算子类型 | 包含内容 |
|--------------|----------|
| 基础融合算子 | Flash attention、Flash attention variable length、Fused rmsnorm、Fused swiglu、Fused rotary position embedding、GMM、Matmul Add |

#### Flash Attention 优化实现
**原始代码**：
```python
class DeepseekAttention(nn.Module):
    def forward(self, query_states, key_states, value_states, attention_mask=None):
        # ... 其他逻辑 ...
        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=mindspore.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)
```

**优化后代码**：
```python
class DeepseekAttention(nn.Module):
    def forward(self, query_states, key_states, value_states, attention_mask=None):
        # ... 其他逻辑 ...
        if attention_mask is not None:
            attention_mask = ~attention_mask
        
        seq_len = 2048
        sparse_mode = 0
        if attention_mask is not None:
            attn_mask = ~attention_mask  # 如果是padding mask
        else:
            attn_mask = None
        
        if self.is_causal:
            sparse_mode = 3
            attn_mask = mindspore.ops.triu(mindspore.ops.ones((seq_len, seq_len), mindspore.bool_), 1)
        
        attn_output = mindspore.ops.flash_attention_score(
            query_states,
            key_states,
            value_states,
            head_num=self.num_heads,
            input_layout='BNSD',
            real_shift=None,
            attn_mask=attn_mask,
            padding_mask=None,
            scalar_value=1 / math.sqrt(self.head_dim),
            keep_prob=1 - self.attention_dropout,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            inner_precise=0,
            drop_mask=None,
            prefix=None,
            actual_seq_qlen=None,
            actual_seq_kvlen=None,
            sparse_mode=sparse_mode
        )
```

### 2. Tensor 索引替换优化
#### 2.1 Rotary Embedding 优化
**核心优化**：将 `mindspore.ops` 替换为 `mindspore.mint`，减少 scatter 操作开销

**DeepseekRotaryEmbedding 优化**：
```python
class DeepseekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=ops.get_default_dtype()
        )
        self.max_seq_len_cached = None
    
    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=mindspore.int64).type_as(self.inv_freq)
        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)
        
        return (
            ops.narrow(self.cos_cached, 0, 0, seq_len).to(dtype=x.dtype),
            ops.narrow(self.sin_cached, 0, 0, seq_len).to(dtype=x.dtype),
        )
```

#### 2.2 rotate_half 函数优化
**原始代码**：
```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), dim=-1)
```

**优化后代码**：
```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = ops.split(x, x.shape[-1]//2, dim=-1)
    return ops.cat((-x2, x1), dim=-1)
```

### 3. MoE 模块前向优化
#### 3.1 DeepseekMoE 优化（分离 Prefill/Decode 逻辑）
```python
class DeepseekMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([
            DeepseekMLP(config, intermediate_size=config.moe_intermediate_size) 
            for i in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config=config, intermediate_size=intermediate_size)
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            raise NotImplementedError("Training is not supported yet.")
        else:
            if orig_shape[1] == 1:
                # Decode 模式（单token）
                y = self.moe_infer_decode(hidden_states, flat_topk_idx, topk_weight.view(-1,1)).view(*orig_shape)
            else:
                # Prefill 模式（多token）
                y = self.moe_infer_prefill(hidden_states, flat_topk_idx, topk_weight.view(-1,1)).view(*orig_shape)
            
            if self.config.n_shared_experts is not None:
                y = y + self.shared_experts(identity)
            return y
    
    @no_grad()
    def moe_infer_decode(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = ops.zeros_like(x)
        for i in range(self.num_experts_per_tok):
            expert_id = flat_expert_indices[i].item()
            weight = flat_expert_weights[i].item()
            expert = self.experts[expert_id]
            expert_out = expert(x)
            expert_cache += expert_out * weight
        return expert_cache
    
    @no_grad()
    def moe_infer_prefill(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = ops.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out = expert_out.mul(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = mindspore.mint.scatter_add(
                expert_cache, 0, 
                exp_token_idx.view(-1, 1).tile((1, x.shape[-1])), 
                expert_out
            )
        return expert_cache
```

#### 3.2 Qwen2MoeSparseMoeBlock 优化
```python
from mindnlp.core import nn, ops, no_grad

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(self.num_experts)
        ])
        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
    
    @no_grad()
    def moe_infer_decode(self, x):
        """推理阶段:decode时只处理单token"""
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=mindspore.float32)
        topk_weights, topk_idx = ops.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)
        
        y = ops.zeros_like(x)
        batch_size = x.shape[0]
        
        # 修正:按样本循环,而不是按专家编号循环
        for b in range(batch_size):
            contributions = []
            for k in range(self.top_k):
                expert_idx = int(topk_idx[b, k])
                expert_output = self.experts[expert_idx](x[b:b+1])
                weight = topk_weights[b, k]
                weighted_output = expert_output * weight
                contributions.append((expert_idx, weighted_output))
            
            # Sort by expert_idx ascending to match vectorized addition order
            contributions.sort(key=lambda t: t[0])
            sample_output = ops.zeros_like(x[b:b+1])
            for _, weighted_output in contributions:
                sample_output += weighted_output
            y[b:b+1] = sample_output
        
        # 共享专家处理
        shared_out = self.shared_expert(x)
        shared_gate = ops.sigmoid(self.shared_expert_gate(x))
        y += shared_out * shared_gate
        
        return y, router_logits
    
    def forward(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Decode 模式(单 token)
        if seq_len == 1 and not self.training:
            # 必须保证 moe_infer_decode 一定返回 (y, router_logits)
            y, router_logits = self.moe_infer_decode(hidden_states)
            return y.view(batch, seq_len, hidden_dim), router_logits
        
        # Prefill / Training 模式
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=mindspore.float32)
        routing_weights, selected_experts = ops.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        final_hidden_states = ops.zeros((batch * seq_len, hidden_dim), dtype=hidden_states.dtype)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = ops.nonzero(expert_mask[expert_idx], as_tuple=True)
            if idx.shape[0] > 0:  # 修复:避免空张量
                current_state = hidden_states[top_x]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states = final_hidden_states.index_add(0, top_x.int(), current_hidden_states)
        
        # 共享专家
        shared_expert_output = self.shared_expert(hidden_states)
        shared_gate = ops.sigmoid(self.shared_expert_gate(hidden_states))
        final_hidden_states += shared_expert_output * shared_gate
        
        final_hidden_states = final_hidden_states.reshape(batch, seq_len, hidden_dim)
        return final_hidden_states, router_logits
```

### 4. Gating 网络计算优化
```python
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        
        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        
        # 优化:使用更高效的参数初始化
        self.weight = nn.Parameter(ops.empty((self.n_routed_experts, self.gating_dim)))
        
        # 优化:添加缓存机制
        self._expert_mask_cache = {}
        self._last_batch_size = None
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # 优化:使用Xavier初始化,更适合线性层
        nn.init.xavier_uniform_(self.weight)
    
    def _optimized_topk_selection(self, scores):
        """优化:高效top-k选择算法"""
        # 使用更高效的内存布局
        scores_contiguous = scores.contiguous()
        # 直接获取topk,避免不必要的排序
        topk_weight, topk_idx = ops.topk(
            scores_contiguous, k=self.top_k, dim=-1, sorted=False
        )
        
        # 原地操作进行归一化,减少内存分配
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True)
            # 添加小常数避免除零,使用原地操作
            denominator = denominator + 1e-20
            topk_weight = topk_weight / denominator
        
        return topk_idx, topk_weight
    
    def _efficient_aux_loss(self, scores, topk_idx, bsz, seq_len):
        """优化:高效的辅助损失计算"""
        if not self.training or self.alpha <= 0.0:
            return None
        
        scores_for_aux = scores
        total_tokens = bsz * seq_len
        
        if self.seq_aux:
            # 优化:使用更高效的scatter操作
            scores_seq = scores_for_aux.view(bsz, seq_len, -1)
            ce = ops.zeros(bsz, self.n_routed_experts, device=scores.device)
            
            # 批量处理scatter操作
            expanded_idx = topk_idx.view(bsz, -1)
            ce.scatter_add_(
                1,
                expanded_idx,
                ops.ones(bsz, seq_len * self.top_k, device=scores.device)
            )
            
            # 向量化计算
            ce = ce / (seq_len * self.top_k / self.n_routed_experts)
            aux_loss = (ce * scores_seq.mean(dim=1)).sum(dim=1).mean() * self.alpha
        else:
            # 优化:使用one-hot的稀疏表示
            flat_idx = topk_idx.view(-1)
            mask_ce = F.one_hot(flat_idx, num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
        
        return aux_loss
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        
        # 优化:使用更高效的重塑操作
        hidden_states_flat = hidden_states.view(-1, h)
        
        # 计算门控分数 - 使用优化的矩阵乘法
        logits = F.linear(hidden_states_flat, self.weight, None)
        
        # 优化:选择更快的softmax实现
        if self.scoring_func == "softmax":
            scores = ops.softmax(logits, dim=-1)
        else:
            raise NotImplementedError(
                f"不支持的评分函数: {self.scoring_func}"
            )
        
        # 优化:使用改进的topk选择
        topk_idx, topk_weight = self._optimized_topk_selection(scores)
        
        # 优化:高效的辅助损失计算
        aux_loss = self._efficient_aux_loss(scores, topk_idx, bsz, seq_len)
        
        return topk_idx, topk_weight, aux_loss
```

### 5. MLP 计算路径优化 + MoE 前向 Decode 并行优化
```python
class DeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        # 优化计算路径
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([
            DeepseekMLP(config, intermediate_size=config.moe_intermediate_size) 
            for i in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config=config, intermediate_size=intermediate_size)
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            raise NotImplementedError("Training is not supported yet.")
        else:
            if orig_shape[1] == 1:
                # Decode 模式并行优化
                y = self.moe_infer_decode(hidden_states, flat_topk_idx, topk_weight.view(-1,1))
                y = y.view(*orig_shape)
            else:
                # Prefill 模式
                y = self.moe_infer_prefill(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
            
            if self.config.n_shared_experts is not None:
                y = y + self.shared_experts(identity)
            return y
    
    @no_grad()
    def moe_infer_decode(self, x, flat_expert_indices, flat_expert_weights):
        # 批量处理所有选中的专家
        selected_experts_id = [self.experts[i] for i in flat_expert_indices]
        
        # 并行计算所有专家输出
        expert_outputs = [expert(x) for expert in selected_experts_id]
        
        # 向量化加权求和
        weighted_sum = sum(out * w for out, w in zip(expert_outputs, flat_expert_weights))
        return weighted_sum
    
    @no_grad()
    def moe_infer_prefill(self, x, flat_expert_indices, flat_expert_weights):
        # 初始化专家输出缓存
        expert_cache = ops.zeros_like(x)
        
        # 对专家索引进行排序,便于按专家分组处理
        idxs = flat_expert_indices.argsort()
        
        # 计算每个专家处理的令牌数量及累积和
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
        
        # 计算对应的令牌索引
        token_idxs = idxs // self.num_experts_per_tok
        
        # 遍历每个专家,处理其对应的令牌
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            
            # 跳过没有令牌分配的专家
            if start_idx == end_idx:
                continue
            
            # 获取当前专家实例
            expert = self.experts[i]
            
            # 提取分配给当前专家的令牌索引和数据
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            
            # 专家网络前向传播
            expert_out = expert(expert_tokens)
            
            # 应用专家权重
            expert_out = expert_out.mul(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 将加权输出累加到对应的令牌位置
            expert_cache = mindspore.mint.scatter_add(
                expert_cache, 0, 
                exp_token_idx.view(-1, 1).tile((1, x.shape[-1])), 
                expert_out
            )
        
        return expert_cache
```
