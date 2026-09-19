# Part 02 · 22 vLLM PagedAttention — 分页缓存管理器模拟实现
import torch
from typing import Dict, List, Tuple


# TODO 1：计算 KV Cache 理论容量
def estimate_kv_cache_bytes(num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> Dict[str, int]:
    """估算 KV Cache 理论容量；不代表 vLLM allocator 的实际峰值。"""
    values = (num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(value <= 0 for value in values):
        raise ValueError('所有容量维度和 dtype_bytes 都必须为正数')
    # 单 token 的 K/V 字节数：K、V 两份 × 层数 × KV 头数 × 头维度 × 每元素字节
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    kv_bytes_per_block = block_size * kv_bytes_per_token
    total_cache_bytes = num_blocks * kv_bytes_per_block
    layout_shape = [num_layers, 2, num_kv_heads, block_size, head_dim]
    return {'kv_bytes_per_token': kv_bytes_per_token, 'kv_bytes_per_block': kv_bytes_per_block,
            'total_cache_bytes': total_cache_bytes, 'layout_shape': layout_shape}


class Request:
    """记录逻辑序列长度，以及逻辑 Block 到物理 Block 的映射。"""

    def __init__(self, request_id: int, prompt_len: int):
        if request_id < 0 or prompt_len <= 0:
            raise ValueError('request_id 必须非负，prompt_len 必须为正数')
        self.request_id = request_id
        self.seq_len = prompt_len
        self.block_table: List[int] = []


class KVCacheManager:
    """用空闲块列表模拟 KV Cache 的分配、扩容、释放和恢复。"""

    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        if num_blocks <= 0 or block_size <= 0 or head_dim <= 0:
            raise ValueError('num_blocks、block_size 和 head_dim 必须为正数')
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        # 模拟预分配的大显存池，形状: [num_blocks, block_size, head_dim]
        self.physical_kv_cache = torch.zeros(num_blocks, block_size, head_dim)
        self.free_blocks: List[int] = list(range(num_blocks))
        # 可选扩展：prefix_key -> {'block_ids': [...], 'refcount': n}
        self.prefix_cache: Dict[Tuple[int, ...], dict] = {}

    def acquire_prefix(self, prefix_tokens: List[int]) -> List[int]:
        """可选扩展：为相同 token 前缀共享物理 Block（引用计数模型）。"""
        key = tuple(int(token) for token in prefix_tokens)
        if not key:
            raise ValueError('prefix_tokens 不能为空')
        entry = self.prefix_cache.get(key)
        if entry is not None:
            entry['refcount'] += 1
            return list(entry['block_ids'])
        needed = (len(key) + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed:
            raise RuntimeError('OOM')
        block_ids = self.free_blocks[:needed]
        del self.free_blocks[:needed]
        self.prefix_cache[key] = {'block_ids': block_ids, 'refcount': 1}
        return list(block_ids)

    def release_prefix(self, prefix_tokens: List[int]):
        key = tuple(int(token) for token in prefix_tokens)
        entry = self.prefix_cache.get(key)
        if entry is None:
            raise KeyError('prefix 不存在或已释放')
        entry['refcount'] -= 1
        if entry['refcount'] == 0:
            self.free_blocks.extend(entry['block_ids'])
            self.free_blocks.sort()
            del self.prefix_cache[key]

    def allocate_for_prefill(self, req: Request):
        """请求刚进来时 (Prefill阶段)，为它的 Prompt 长度分配所需的全部 Block。"""
        if req.block_table:
            raise ValueError('请求已经完成 Prefill，不能重复分配')
        # TODO 2: 计算需要的 block 数量（向上取整）
        needed_blocks = (req.seq_len + self.block_size - 1) // self.block_size
        # TODO 3: 先确认空闲块足够，再一次性提交块表，避免 OOM 后半分配
        if len(self.free_blocks) < needed_blocks:
            raise RuntimeError("OOM")
        allocated_blocks = self.free_blocks[:needed_blocks]
        del self.free_blocks[:needed_blocks]
        req.block_table.extend(allocated_blocks)

    def allocate_for_decode(self, req: Request):
        """Decode 阶段：跨过 Block 边界时才按需分配 1 个新 Block。"""
        new_seq_len = req.seq_len + 1
        # TODO 4: 加 1 后落在块边界开头（余数为 1）说明要跨入新块
        is_new_block_needed = (new_seq_len % self.block_size) == 1
        if is_new_block_needed:
            # OOM 时不能修改 seq_len 或 block_table，先检查资源
            if not self.free_blocks:
                raise RuntimeError("OOM")
            req.block_table.append(self.free_blocks.pop(0))
        req.seq_len = new_seq_len

    # TODO 5：计算请求的 Block 占用报告
    def allocation_report(self, req: Request) -> Dict[str, float]:
        allocated_tokens = len(req.block_table) * self.block_size
        unused_tail_tokens = allocated_tokens - req.seq_len
        utilization = req.seq_len / allocated_tokens if allocated_tokens else 0.0
        return {'logical_tokens': req.seq_len, 'allocated_tokens': allocated_tokens,
                'unused_tail_tokens': unused_tail_tokens, 'utilization': utilization}

    def release_request(self, req: Request):
        """释放请求占用的物理块，并清空其块表。"""
        # TODO 6: 先校验再归还，防止重复/越界块污染空闲池
        released_block_ids = list(req.block_table)
        if not released_block_ids:
            raise ValueError('请求没有可释放的物理块，可能已经释放')
        if len(set(released_block_ids)) != len(released_block_ids):
            raise ValueError('block_table 不能包含重复物理块')
        if any(b < 0 or b >= self.num_blocks for b in released_block_ids):
            raise ValueError('请求包含越界物理块')
        if any(b in self.free_blocks for b in released_block_ids):
            raise ValueError('请求包含已释放的物理块')
        self.free_blocks.extend(released_block_ids)
        self.free_blocks.sort()
        req.block_table.clear()

    # TODO 7：恢复逻辑连续 Cache
    def get_physical_cache(self, req: Request) -> torch.Tensor:
        """根据块表恢复逻辑连续的 KV Cache。"""
        if not req.block_table:
            raise ValueError('请求没有可读取的物理块')
        if any(b < 0 or b >= self.num_blocks for b in req.block_table):
            raise ValueError('block_table 包含越界物理块')
        # 必须按 block_table 的逻辑顺序读取，不能按物理索引排序
        blocks = [self.physical_kv_cache[b] for b in req.block_table]
        cat_blocks = torch.cat(blocks, dim=0)
        return cat_blocks[:req.seq_len]


def run_prefix_cache_extension_check():
    """扩展测试：验证相同前缀的 Block 共享和引用计数释放。"""
    manager = KVCacheManager(num_blocks=4, block_size=4, head_dim=8)
    prefix = [101, 102, 103, 104, 105]
    first_ids = manager.acquire_prefix(prefix)
    second_ids = manager.acquire_prefix(prefix)
    assert first_ids == second_ids, '相同 prefix 应共享同一组物理 Block'
    assert len(manager.free_blocks) == 2, 'Prefix Cache 命中不应重复申请 Block'
    manager.release_prefix(prefix)
    assert len(manager.free_blocks) == 2, '仍有引用时不能释放共享 Block'
    manager.release_prefix(prefix)
    assert manager.free_blocks == [0, 1, 2, 3], '引用计数归零后应归还 Block'
    print('✅ Prefix Cache 扩展测试通过！')
