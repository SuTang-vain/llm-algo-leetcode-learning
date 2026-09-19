# Part 02 · 21 解码策略 — Temperature / Top-k / Top-p 实现
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """在 Softmax 前缩放 logits；temperature 必须是有限正数。"""
    # TODO 1: 检查 temperature 并完成缩放
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError('temperature 必须是有限正数')
    temp = float(temperature)
    return logits / temp


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """按第 K 大值筛选 logits，保留 ties 是明确语义。"""
    if top_k is None:
        return logits
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError('top_k 必须为非负整数')
    if top_k == 0 or top_k >= logits.size(-1):
        return logits
    # TODO 2: Top-K 截断（阈值语义：不小于第 K 大阈值的全保留）
    filter_value = float('-inf')
    kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    logits = torch.where(logits < kth_values, torch.full_like(logits, filter_value), logits)
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """按累计概率保留最小候选集合，并恢复原始词表顺序（首次达到阈值语义）。"""
    if top_p is None or top_p == 1.0:
        return logits
    if not 0.0 < top_p < 1.0:
        raise ValueError('top_p 必须满足 0 < top_p <= 1')
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # TODO 3: Top-p 核心逻辑
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # 至少保留概率最大的 token
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    restored_logits = torch.empty_like(logits)
    restored_logits.scatter_(-1, sorted_indices, sorted_logits)
    return restored_logits


def decode_next_token(logits: torch.Tensor, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, generator=None):
    """完成一次候选过滤和 token 选择，支持 greedy 与可复现 sampling。"""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError('logits 必须是 [vocab] 或 [batch, vocab]')
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_token


def autoregressive_decode(prompt_ids, logits_fn, max_new_tokens, **decode_kwargs):
    """用 logits_fn 演示逐 token 解码；不代表真实模型性能。"""
    tokens = prompt_ids.clone()
    for _ in range(max_new_tokens):
        step_logits = logits_fn(tokens)
        if step_logits.dim() == 3:
            step_logits = step_logits[:, -1, :]
        next_token = decode_next_token(step_logits, **decode_kwargs)
        tokens = torch.cat([tokens, next_token], dim=-1)
    return tokens
