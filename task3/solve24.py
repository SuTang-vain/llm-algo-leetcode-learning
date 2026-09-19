# Part 02 · 24 SGLang RadixAttention — 压缩前缀树模拟实现
import torch


class TreeNode:
    """表示一条压缩边及其子路径；`terminal` 表示完整路径在此结束。"""

    def __init__(self, key_tokens, terminal=False):
        self.key_tokens = list(key_tokens)  # 这条边上的 Token 序列
        self.children = []            # 子节点列表
        self.kv_cache_ptr = None      # 模拟指向物理 KV Cache 的指针
        self.terminal = terminal      # 是否有完整请求在此结束


class SimpleRadixCache:
    """用压缩 Radix Tree 模拟公共 token 前缀的登记与匹配。"""

    def __init__(self):
        self.root = TreeNode([])

    def _find_child(self, node, token):
        """返回首 token 匹配的子边；同一节点下不应出现重复首 token。"""
        return next((c for c in node.children if c.key_tokens and c.key_tokens[0] == token), None)

    def insert(self, tokens):
        """插入路径；遇到部分重叠边时分裂旧边。"""
        tokens = list(tokens)
        if not tokens:
            raise ValueError('tokens 不能为空')
        node, offset = self.root, 0
        while offset < len(tokens):
            # TODO 1: 按首 token 找候选子边，并计算公共边长度
            child = self._find_child(node, tokens[offset])
            if child is None:
                # 没有同首 token 的 child：新增叶节点并标记 terminal，
                # 这条完整路径已登记，可作为后续复用候选
                node.children.append(TreeNode(tokens[offset:], terminal=True))
                return
            common = self._lcp_len(child.key_tokens, tokens[offset:])
            if common == len(child.key_tokens):
                node, offset = child, offset + common
                continue
            # TODO 2: 用 shared 边替换 child，并挂接旧后缀和新后缀
            shared = TreeNode(child.key_tokens[:common])
            old_suffix = TreeNode(child.key_tokens[common:], terminal=child.terminal)
            old_suffix.children = child.children
            old_suffix.kv_cache_ptr = child.kv_cache_ptr
            shared.children.append(old_suffix)
            node.children[node.children.index(child)] = shared
            if offset + common == len(tokens):
                # 新路径是旧路径的前缀：shared 节点本身成为 terminal
                shared.terminal = True
            else:
                shared.children.append(TreeNode(tokens[offset + common:], terminal=True))
            return

    def _lcp_len(self, cached_tokens, prompt_tokens):
        """计算两段 token 序列的最长公共前缀长度。"""
        # TODO 3: 逐个 token 比较，遇到不相等时立刻停止
        match_len = 0
        while (match_len < len(cached_tokens) and match_len < len(prompt_tokens)
               and cached_tokens[match_len] == prompt_tokens[match_len]):
            match_len += 1
        return match_len

    def match_prefix(self, prompt_tokens):
        """为新 prompt 寻找最长匹配前缀：前 N 个 token 一致则其 KV Cache 可直接复用。"""
        prompt_tokens = list(prompt_tokens)
        node, offset, best_match_len = self.root, 0, 0
        while offset < len(prompt_tokens):
            # TODO 4: 沿首 token 对应的 child 向下查找完整共享边
            child = self._find_child(node, prompt_tokens[offset])
            if child is None:
                break
            common = self._lcp_len(child.key_tokens, prompt_tokens[offset:])
            if common < len(child.key_tokens):
                # 边只命中一部分：这条缓存路径不完整，不能复用
                break
            offset += common
            node = child
            # TODO 5: 只有终止节点才算完整可复用前缀
            if node.terminal:
                best_match_len = offset
        return best_match_len

    def split_prompt(self, prompt_tokens):
        """把 prompt 拆成可复用前缀和需要重算的后缀。"""
        # TODO 6: 先找命中长度，再拆出前缀和后缀
        hit_len = self.match_prefix(prompt_tokens)
        hit_prefix = prompt_tokens[:hit_len]
        miss_suffix = prompt_tokens[hit_len:]
        return hit_prefix, miss_suffix, hit_len
