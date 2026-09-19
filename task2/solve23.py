# Part 02 · 23 投机解码 — 接受/拒绝 + residual correction + bonus token
import torch


def speculative_decode_step(draft_probs, target_probs, draft_tokens, generator=None):
    """
    验证 K 个草稿，并在拒绝或全接受时生成修正 token。

    Args:
        draft_probs: 草稿模型概率分布, shape [K, vocab_size]
        target_probs: 目标模型概率分布（含 1 个 bonus 位置）, shape [K+1, vocab_size]
        draft_tokens: 草稿 token_id, shape [K]

    Returns:
        dict: tokens / accepted_count / rejected / target_positions_checked
    """
    if draft_probs.dim() != 2 or target_probs.dim() != 2:
        raise ValueError('概率张量必须是二维 [K, vocab]')
    K, vocab_size = draft_probs.shape
    if target_probs.shape != (K + 1, vocab_size) or len(draft_tokens) != K:
        raise ValueError('draft/target/token 的长度或词表维度不匹配')
    if not (torch.isfinite(draft_probs).all() and torch.isfinite(target_probs).all()):
        raise ValueError('概率分布不能包含 NaN 或 Inf')
    # TODO 1: 检查概率非负，并验证每一行和约等于 1
    if (draft_probs < 0).any() or (target_probs < 0).any():
        raise ValueError('输入必须是归一化概率分布')
    ones = torch.ones(vocab_size, device=draft_probs.device, dtype=draft_probs.dtype)
    if not torch.allclose(draft_probs @ ones, torch.ones(K, device=draft_probs.device, dtype=draft_probs.dtype), atol=1e-5):
        raise ValueError('输入必须是归一化概率分布')
    if not torch.allclose(target_probs @ ones, torch.ones(K + 1, device=target_probs.device, dtype=target_probs.dtype), atol=1e-5):
        raise ValueError('输入必须是归一化概率分布')

    accepted_tokens = []

    for i in range(K):
        token_id = draft_tokens[i]
        if not 0 <= int(token_id) < vocab_size:
            raise ValueError('draft token id 超出词表范围')
        p = target_probs[i, token_id]
        q = draft_probs[i, token_id]

        # TODO 2: 接受概率 alpha = min(1, p/q)；q=0 时先处理除零边界
        r = torch.rand((), generator=generator).item()
        if q == 0:
            accept = p > 0  # q=0 且 p>0 直接接受；p 也为 0 进入拒绝分支
        else:
            accept = r < min(1.0, (p / q).item())
        if accept:
            accepted_tokens.append(int(token_id))
            continue
        # TODO 3: 拒绝时从 residual = max(target - draft, 0) 归一化后采样
        residual = (target_probs[i] - draft_probs[i]).clamp_min(0)
        residual_mass = residual.sum()
        if residual_mass <= 0:
            raise ValueError('residual 概率质量必须大于 0')
        residual = residual / residual_mass
        correction = torch.multinomial(residual, 1, generator=generator).item()
        return {'tokens': accepted_tokens + [correction], 'accepted_count': len(accepted_tokens),
                'rejected': True, 'target_positions_checked': i + 1}

    # TODO 4: 全部接受后，从 target_probs[K] 采样 bonus token
    bonus = torch.multinomial(target_probs[K], 1, generator=generator).item()
    return {'tokens': accepted_tokens + [bonus], 'accepted_count': K,
            'rejected': False, 'target_positions_checked': K}
