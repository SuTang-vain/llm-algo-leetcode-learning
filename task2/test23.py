from solve23 import *

def test_speculative_decoding():
    try:
        torch.manual_seed(42)
        vocab_size = 5
        K = 3
        
        # 模拟生成
        draft_tokens = [0, 1, 2]
        draft_probs = torch.zeros(K, vocab_size)
        target_probs = torch.zeros(K + 1, vocab_size)
        draft_probs[0, 0] = 1.0; target_probs[0, 0] = 1.0  # 必接受
        draft_probs[1, 1] = 1.0; target_probs[1, 3] = 1.0  # 必拒绝，residual 采样 3
        draft_probs[2, 2] = 1.0; target_probs[2, 2] = 1.0
        target_probs[3, 4] = 1.0  # 全部接受时的 bonus token
        
        result = speculative_decode_step(draft_probs, target_probs, draft_tokens)
        assert result['tokens'] == [0, 3] and result['accepted_count'] == 1 and result['rejected']
        all_accepted = speculative_decode_step(target_probs[:K], target_probs, [0, 3, 2])
        assert all_accepted['tokens'] == [0, 3, 2, 4] and not all_accepted['rejected']

        # q=0 且 p>0：候选仍可直接接受，避免除零后错误拒绝
        zero_q_draft = torch.tensor([[0.0, 1.0, 0.0]])
        zero_q_target = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
        zero_q = speculative_decode_step(zero_q_draft, zero_q_target, [0])
        assert zero_q['accepted_count'] == 1 and not zero_q['rejected']

        # 非法概率必须在进入接受逻辑前被拒绝
        invalid = zero_q_draft.clone()
        invalid[0, 0] = -0.1
        try:
            speculative_decode_step(invalid, zero_q_target, [0])
        except ValueError:
            pass
        else:
            raise AssertionError('负概率应明确被拒绝')

        # 草稿提出了 q=0 且目标也为 0 的不可能 token，residual 没有可采样质量
        zero_residual_draft = torch.tensor([[1.0, 0.0]])
        zero_residual_target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        try:
            speculative_decode_step(zero_residual_draft, zero_residual_target, [1])
        except ValueError:
            pass
        else:
            raise AssertionError('residual 概率质量为 0 时应明确报错')
        print("✅ 测试通过！接受、residual correction 和 bonus token 均通过。")
        
    except NotImplementedError:
        print("请先完成 TODO 代码。")
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
        elif isinstance(e, AssertionError):
            print("代码可能未完成，导致了断言失败")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_speculative_decoding()
