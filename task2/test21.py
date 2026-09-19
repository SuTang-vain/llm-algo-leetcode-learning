from solve21 import *

# 运行此单元格以测试你的实现
def test_decoding():
    try:
        # 为了保证可重复性
        torch.manual_seed(42)
        vocab_size = 10
        # 伪造一组 Logits: [0.1, 2.3, 0.4, 1.2, -0.5, 4.0, 3.1, 0.0, 1.1, -1.0]
        # 最大的两个: index 5 (4.0), index 6 (3.1)
        logits = torch.tensor([[0.1, 2.3, 0.4, 1.2, -0.5, 4.0, 3.1, 0.0, 1.1, -1.0]])
        
        print("原始 Logits (前10个单词):", logits.squeeze().tolist())
        
        # 1. 测试 Temperature
        t_logits = apply_temperature(logits.clone(), 0.5)
        # 温度 0.5 应该会让差异翻倍
        assert torch.allclose(t_logits[0, 5] - t_logits[0, 6], (logits[0, 5] - logits[0, 6]) * 2), "温度调节错误！"
        assert t_logits.shape == logits.shape and t_logits.dtype == logits.dtype and t_logits.device == logits.device, "温度调节不应改变张量接口"
        print("✅ Temperature 温度调节通过！")
        
        # 2. 测试 Top-K
        k_logits = apply_top_k(logits.clone(), 3)
        # 只保留最大的三个：5, 6, 1
        valid_count = (k_logits != float('-inf')).sum().item()
        assert valid_count == 3, f"Top-K 截断没有正确执行，保留了 {valid_count} 个值"
        print("✅ Top-K 暴力截断通过！")
        
        # 3. 测试 Top-p
        # 原始概率: [0.01, 0.10, 0.01, 0.03, 0.00, 0.54, 0.22, 0.01, 0.03, 0.00]
        # 降序: 0.54 (idx 5), 0.22 (idx 6), 0.10 (idx 1) ...
        # 累加和: 0.54, 0.76, 0.86
        # 所以只有 idx 5, 6, 1 会被保留
        p_logits = apply_top_p(logits.clone(), 0.8)
        valid_count = (p_logits != float('-inf')).sum().item()
        assert valid_count == 3, f"Top-p 核采样截断没算准，保留了 {valid_count} 个值"
        assert torch.equal(p_logits[0, [5, 6, 1]], logits[0, [5, 6, 1]]), "Top-p 不应打乱原词表顺序"
        print("✅ Top-p (Nucleus) 核采样动态截断通过！")

        # 边界：Top-k 采用阈值语义，因此 ties 可能保留超过 k 个候选
        tied = torch.tensor([[2.0, 2.0, 2.0, 1.0]])
        tied_result = apply_top_k(tied, 2)
        assert torch.isfinite(tied_result).sum().item() == 3, "Top-k ties 的阈值语义不一致"
        for invalid_p in (0.0, 1.1):
            try:
                apply_top_p(logits, invalid_p)
                raise AssertionError('非法 top_p 未被拒绝')
            except ValueError:
                pass
        
        # 4. 测试完整管线
        next_token = decode_next_token(logits.clone(), temperature=0.7, top_k=50, top_p=0.9)
        assert next_token.shape == (1, 1), "解码的词张量维度不对"

        # 5. greedy、参数边界与最小自回归循环
        greedy = decode_next_token(logits, temperature=1.0, top_k=0, top_p=1.0, do_sample=False)
        assert greedy.item() == 5, "greedy 应选择最大 logits 的索引"
        one_dim_greedy = decode_next_token(logits.squeeze(0), temperature=1.0, top_k=0, top_p=1.0, do_sample=False)
        assert one_dim_greedy.shape == (1, 1), "一维 logits 应统一返回 [batch, 1]"
        assert torch.equal(torch.argsort(logits, dim=-1), torch.argsort(t_logits, dim=-1)), "temperature 不应改变排序"
        try:
            apply_temperature(logits, 0.0)
            raise AssertionError('非法 temperature 未被拒绝')
        except ValueError:
            pass

        prompt = torch.tensor([[1, 2]])
        def fake_logits(tokens):
            output = torch.zeros(tokens.size(0), tokens.size(1), vocab_size)
            output[..., 3] = 2.0
            return output
        generated = autoregressive_decode(prompt, fake_logits, max_new_tokens=3, temperature=1.0, top_k=0, top_p=1.0, do_sample=False)
        assert generated.shape == (1, 5) and torch.equal(generated[0, -3:], torch.tensor([3, 3, 3])), "自回归循环结果错误"

        # 6. batch 输入和随机种子：检查张量接口与 sampling 可复现性
        batch_logits = logits.repeat(2, 1)
        assert apply_top_k(batch_logits, 3).shape == batch_logits.shape
        assert apply_top_p(batch_logits, 0.8).shape == batch_logits.shape
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        sample_1 = decode_next_token(logits, generator=g1)
        sample_2 = decode_next_token(logits, generator=g2)
        assert torch.equal(sample_1, sample_2), "相同随机种子应得到相同采样结果"

        # 7. 用候选数和熵观察策略差异；这不是质量或吞吐结论
        for name, kwargs in {
            'greedy': {'temperature': 1.0, 'top_k': 0, 'top_p': 1.0},
            'top_k': {'temperature': 1.0, 'top_k': 3, 'top_p': 1.0},
            'top_p': {'temperature': 1.0, 'top_k': 0, 'top_p': 0.8},
        }.items():
            filtered = apply_temperature(logits.clone(), kwargs['temperature'])
            filtered = apply_top_k(filtered, kwargs['top_k'])
            filtered = apply_top_p(filtered, kwargs['top_p'])
            probs = F.softmax(filtered, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            print(f'{name}: candidates={(torch.isfinite(filtered)).sum().item()}, entropy={entropy.item():.4f}')
        print(f"\n✅ All Tests Passed! 解码策略实现通过测试。本次采样的下一个 token ID 是: {next_token.item()}")
        
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了张量维度错误")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_decoding()
