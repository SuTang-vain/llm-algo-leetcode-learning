from solve22 import *

# 运行此单元格以测试你的实现
def test_paged_attention_manager():
    try:
        # Case 1: 典型 Prefill + Decode + Cache 拼装
        manager = KVCacheManager(num_blocks=10, block_size=4, head_dim=64)
        print("初始化内存池...")

        for bad_request in ((-1, 2), (1, 0)):
            try:
                Request(request_id=bad_request[0], prompt_len=bad_request[1])
                raise AssertionError('非法请求参数未被拒绝')
            except ValueError:
                pass
        print("✅ Request 输入校验通过！")

        req1 = Request(request_id=1, prompt_len=6)
        manager.allocate_for_prefill(req1)
        assert len(req1.block_table) == 2, "长度 6 的请求应分配 2 个 Block！"
        assert len(manager.free_blocks) == 8, "池中应该剩下 8 个空闲块！"
        print(f"✅ Prefill 测试通过！Req1 分配的块表: {req1.block_table}")

        try:
            manager.allocate_for_prefill(req1)
        except ValueError:
            print("✅ 重复 Prefill 已拒绝！")
        else:
            raise AssertionError('同一请求不能重复 Prefill 并追加 Block')

        manager.allocate_for_decode(req1)
        assert len(req1.block_table) == 2, "生成第 7 个 token 时不应该分配新块！"

        manager.allocate_for_decode(req1)
        manager.allocate_for_decode(req1)
        assert len(req1.block_table) == 3, "生成第 9 个 token 时应当分配了第 3 个新块！"
        assert len(manager.free_blocks) == 7, "池中应该剩下 7 个空闲块！"
        print(f"✅ Decode 动态分配测试通过！Req1 最新块表: {req1.block_table}")

        for block_id, value in zip(req1.block_table, [1.0, 2.0, 3.0]):
            manager.physical_kv_cache[block_id].fill_(value)
        cache = manager.get_physical_cache(req1)
        assert cache.shape == (9, 64), f"拼装出来的连续 Cache 形状不对，应为 (9, 64)，实为 {cache.shape}"
        assert torch.all(cache[:4] == 1.0), "第 1 个 Block 未正确拼装！"
        assert torch.all(cache[4:8] == 2.0), "第 2 个 Block 未正确拼装！"
        assert torch.all(cache[8:] == 3.0), "第 3 个 Block 的截断拼装不正确！"
        print("✅ Cache 拼装测试通过！多块物理缓存被正确恢复为逻辑连续序列。")

        # Case 1b: 物理块可以不连续，但恢复必须遵循逻辑 Block 顺序
        scattered_manager = KVCacheManager(num_blocks=4, block_size=2, head_dim=2)
        scattered_req = Request(request_id=10, prompt_len=3)
        scattered_req.block_table = [3, 1]
        scattered_manager.physical_kv_cache[3].fill_(3.0)
        scattered_manager.physical_kv_cache[1].fill_(1.0)
        scattered_cache = scattered_manager.get_physical_cache(scattered_req)
        assert torch.all(scattered_cache[:2] == 3.0)
        assert torch.all(scattered_cache[2:] == 1.0)
        print("✅ 非连续物理块按逻辑顺序恢复通过！")

        # Case 2: 恰好跨越 block 边界时，Decode 应该分配新块，并正确截断最后一块
        manager2 = KVCacheManager(num_blocks=4, block_size=4, head_dim=8)
        req2 = Request(request_id=2, prompt_len=4)
        manager2.allocate_for_prefill(req2)
        assert len(req2.block_table) == 1, "长度 4 的请求应只分配 1 个 Block！"
        manager2.allocate_for_decode(req2)
        assert len(req2.block_table) == 2, "长度 5 的请求应分配第 2 个 Block！"
        manager2.physical_kv_cache[req2.block_table[0]].fill_(7.0)
        manager2.physical_kv_cache[req2.block_table[1]].fill_(8.0)
        cache2 = manager2.get_physical_cache(req2)
        assert cache2.shape == (5, 8), f"拼装出来的连续 Cache 形状不对，应为 (5, 8)，实为 {cache2.shape}"
        assert torch.all(cache2[:4] == 7.0), "边界块的前 4 个 token 不正确！"
        assert torch.all(cache2[4:] == 8.0), "边界块的最后 1 个 token 不正确！"
        print("✅ 边界分配与截断测试通过！")

        # Case 3: OOM 分支必须抛出 RuntimeError
        oom_manager = KVCacheManager(num_blocks=1, block_size=4, head_dim=8)
        oom_req = Request(request_id=3, prompt_len=5)
        try:
            oom_manager.allocate_for_prefill(oom_req)
        except RuntimeError as e:
            assert "OOM" in str(e), "OOM 异常信息不正确！"
            print("✅ OOM 测试通过！")
        else:
            raise AssertionError('显存池不足时应该抛出 RuntimeError("OOM")！')
        # Case 4: Prefill OOM 不应留下半分配状态
        atomic_manager = KVCacheManager(num_blocks=2, block_size=4, head_dim=8)
        atomic_req = Request(request_id=4, prompt_len=9)
        free_before = list(atomic_manager.free_blocks)
        try:
            atomic_manager.allocate_for_prefill(atomic_req)
        except RuntimeError as e:
            assert 'OOM' in str(e), '应明确报告 OOM'
            assert atomic_req.block_table == [], 'OOM 后请求不应保留半分配块'
            assert atomic_manager.free_blocks == free_before, 'OOM 后空闲池不应被部分消耗'
            print('✅ Prefill 原子 OOM 测试通过！')
        else:
            raise AssertionError('资源不足时必须抛出 OOM')

        ledger = estimate_kv_cache_bytes(num_blocks=4, block_size=4, num_layers=2, num_kv_heads=8, head_dim=16, dtype_bytes=2)
        assert ledger['kv_bytes_per_token'] == 1024, 'K/V 两份缓存账本计算不正确'
        assert ledger['total_cache_bytes'] == 16384, 'Block 池总容量计算不正确'
        reusable_manager = KVCacheManager(num_blocks=3, block_size=4, head_dim=8)
        reusable_req = Request(request_id=5, prompt_len=5)
        reusable_manager.allocate_for_prefill(reusable_req)
        allocated_before_release = list(reusable_req.block_table)
        report = reusable_manager.allocation_report(reusable_req)
        assert report['allocated_tokens'] == 8, '已分配 token 账本不正确'
        assert report['unused_tail_tokens'] == 3, '尾块浪费计算不正确'
        assert round(report['utilization'], 3) == 0.625, 'Block 利用率计算不正确'
        reusable_manager.release_request(reusable_req)
        assert reusable_req.block_table == [], '释放后请求块表应为空'
        assert reusable_manager.free_blocks == [0, 1, 2], '释放后的 Block 应可复用'
        try:
            reusable_manager.release_request(reusable_req)
        except ValueError:
            pass
        else:
            raise AssertionError('重复释放请求应明确报错')
        rollback_manager = KVCacheManager(num_blocks=1, block_size=4, head_dim=8)
        rollback_req = Request(request_id=6, prompt_len=4)
        rollback_manager.allocate_for_prefill(rollback_req)
        old_seq_len = rollback_req.seq_len
        try:
            rollback_manager.allocate_for_decode(rollback_req)
        except RuntimeError:
            assert rollback_req.seq_len == old_seq_len, 'Decode OOM 后 seq_len 不应改变'
            print('✅ Decode OOM 回滚测试通过！')
        else:
            raise AssertionError('Decode 跨块时应触发 OOM')
        print('✅ KV Cache 账本与释放复用测试通过！')

        print("\n✅ All Tests Passed! PagedAttention 内存管理逻辑验证通过。")

    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except Exception as e:
        # 只把 TODO 未完成单独提示；断言、类型和状态错误保留原始异常，便于定位。
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        raise


test_paged_attention_manager()

run_prefix_cache_extension_check()
