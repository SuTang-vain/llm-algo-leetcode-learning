# Part 02 · 20 FlashAttention Sim — CPU 分块模拟实现
import torch
import math


def flash_attention_forward_sim(q, k, v, block_size=2, causal=False):
    """计算二维输入上的 FlashAttention 前向模拟。

    Args:
        q, k, v: [seq_len, dim] 张量，device 和 dtype 应保持一致。
        block_size: Q/K/V 的分块大小，必须为正数。
        causal: 是否只允许关注当前位置及之前的 K token。

    Returns:
        [seq_len, dim] 的 attention 输出。
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError('q、k、v 必须是 [seq_len, dim] 二维张量')
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError('q、k、v 的形状必须一致')
    if q.device != k.device or k.device != v.device:
        raise ValueError('q、k、v 必须位于同一 device')
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise TypeError('q、k、v 必须使用相同 dtype')
    if block_size <= 0:
        raise ValueError('block_size 必须为正数')

    seq_len, dim = q.shape

    # TODO 1: 初始化输出 O，全局最大值 m，全局指数和 l
    out = torch.zeros((seq_len, dim), device=q.device, dtype=q.dtype)
    m = torch.full((seq_len, 1), -float('inf'), device=q.device, dtype=q.dtype)
    l = torch.zeros((seq_len, 1), device=q.device, dtype=q.dtype)

    scale = 1.0 / math.sqrt(dim)

    # 外层循环：遍历 Q 的分块
    for i in range(0, seq_len, block_size):
        q_block = q[i:i+block_size] * scale
        m_i = m[i:i+block_size]
        l_i = l[i:i+block_size]
        out_i = out[i:i+block_size]

        # 内层循环：遍历 K, V 的分块
        for j in range(0, seq_len, block_size):
            k_block = k[j:j+block_size]
            v_block = v[j:j+block_size]

            # TODO 2: 计算当前 Q/K block 的缩放 score S_ij
            S_ij = q_block @ k_block.transpose(-2, -1)
            # TODO 2a（可选 causal mask）：屏蔽 key_pos > query_pos 的 score
            if causal:
                query_pos = torch.arange(i, i + q_block.shape[0], device=q.device)[:, None]
                key_pos = torch.arange(j, j + k_block.shape[0], device=q.device)[None, :]
                S_ij = S_ij.masked_fill(key_pos > query_pos, -float('inf'))

            # TODO 3: 局部最大值 m_block 与新的全局最大值 m_new
            m_block = torch.max(S_ij, dim=-1, keepdim=True)[0]
            m_new = torch.maximum(m_i, m_block)

            # TODO 4: 未归一化的指数权重（以 m_new 为基准，保证数值稳定）
            exp_scores = torch.exp(S_ij - m_new)

            # TODO 5: 局部指数和 l_block，并按修正公式更新全局指数和 l_new
            l_block = torch.sum(exp_scores, dim=-1, keepdim=True)
            l_new = l_i * torch.exp(m_i - m_new) + l_block

            # TODO 6: 更新输出 O_i（旧状态按旧分母占比缩放，再加当前块贡献）
            out_i = out_i * (l_i * torch.exp(m_i - m_new) / l_new) + (exp_scores @ v_block) / l_new

            # 更新全局状态
            m_i = m_new
            l_i = l_new

        # 写回全局变量
        out[i:i+block_size] = out_i
        m[i:i+block_size] = m_i
        l[i:i+block_size] = l_i

    return out


def test_flash_attention_sim():
    try:
        import math

        def run_case(seq_len, dim, block_size, seed):
            torch.manual_seed(seed)
            q = torch.randn(seq_len, dim)
            k = torch.randn(seq_len, dim)
            v = torch.randn(seq_len, dim)

            scale = 1.0 / math.sqrt(dim)
            scores = (q @ k.transpose(-2, -1)) * scale
            attn = torch.nn.functional.softmax(scores, dim=-1)
            out_ref = attn @ v

            out_sim = flash_attention_forward_sim(q, k, v, block_size=block_size)
            diff = torch.max(torch.abs(out_ref - out_sim))
            print(f"[seq={seq_len}, dim={dim}, block={block_size}] 最大误差: {diff.item():.6e}")
            assert diff < 1e-5, f"计算结果与标准 Attention 不一致！(seq={seq_len}, dim={dim}, block={block_size})"

        run_case(seq_len=8, dim=4, block_size=2, seed=42)
        run_case(seq_len=5, dim=3, block_size=3, seed=7)
        run_case(seq_len=3, dim=2, block_size=1, seed=123)

        # causal 扩展测试
        torch.manual_seed(11)
        q = torch.randn(6, 4)
        k = torch.randn(6, 4)
        v = torch.randn(6, 4)
        causal_scores = (q @ k.transpose(-2, -1)) / math.sqrt(4)
        causal_scores = causal_scores.masked_fill(torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1), -float('inf'))
        causal_ref = torch.softmax(causal_scores, dim=-1) @ v
        causal_out = flash_attention_forward_sim(q, k, v, block_size=2, causal=True)
        assert torch.allclose(causal_ref, causal_out, atol=1e-5, rtol=1e-5)

        # dtype 边界
        torch.manual_seed(9)
        q64 = torch.randn(4, 3, dtype=torch.float64)
        k64 = torch.randn(4, 3, dtype=torch.float64)
        v64 = torch.randn(4, 3, dtype=torch.float64)
        ref64 = torch.softmax((q64 @ k64.transpose(-2, -1)) / math.sqrt(3), dim=-1) @ v64
        out64 = flash_attention_forward_sim(q64, k64, v64, block_size=2)
        assert out64.dtype == q64.dtype
        assert torch.allclose(ref64, out64, atol=1e-10, rtol=1e-10)

        # 工作集观察
        seq_len = 128
        full_score_elements = seq_len * seq_len
        for block_size in [1, 4, 16, 32]:
            tile_score_elements = block_size * block_size
            print(f"block={block_size}: 完整 score={full_score_elements}; 单 tile={tile_score_elements}")
            assert tile_score_elements < full_score_elements

        # 数值稳定性
        q_large = torch.full((3, 2), 100.0)
        k_large = torch.full((3, 2), 100.0)
        v_large = torch.randn(3, 2)
        stable_out = flash_attention_forward_sim(q_large, k_large, v_large, block_size=2)
        assert torch.isfinite(stable_out).all(), 'online softmax 应保持有限输出'

        try:
            flash_attention_forward_sim(torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2), block_size=0)
        except ValueError:
            pass
        else:
            raise AssertionError('block_size <= 0 应该被拒绝')

        print("✅ Online Softmax 与分块计算逻辑正确！")
        print("\n FlashAttention 分块计算逻辑验证通过。")

    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except NameError as exc:
        print("题目区仍有变量未完成，请根据 TODO 补全实现。")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from exc
    except (AttributeError, TypeError, ValueError, AssertionError, RuntimeError):
        raise
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Part 02 · 20 FlashAttention Sim | CPU 运行")
    print("=" * 60)
    test_flash_attention_sim()
