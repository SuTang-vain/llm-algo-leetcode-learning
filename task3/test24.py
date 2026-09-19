from solve24 import *

# 测试你的实现
def test_radix_attention():
    try:
        cache = SimpleRadixCache()
        cache.insert([0, 1, 2, 3])
        cache.insert([0, 1, 2, 3, 4])
        cache.insert([9, 9, 9])

        # 1. 基础 LCP 检查
        assert cache._lcp_len([1, 2, 3], [1, 2, 4]) == 2, "LCP 计算失败！"
        assert cache._lcp_len([7, 8], [7, 8, 9, 10]) == 2, "完整前缀匹配失败！"
        print("✅ 最长公共前缀计算正确！")

        # 2. 多候选路径下，应该选择最长命中前缀
        match_len = cache.match_prefix([0, 1, 2, 3, 4, 5])
        assert match_len == 5, "匹配失败！应该命中最长的 5 个 token 前缀。"
        assert cache.match_prefix([9, 9, 9]) == 3, "新建完整路径必须标记为 terminal！"
        assert cache.match_prefix([7, 6, 5]) == 0, "错误匹配！不该匹配到任何东西。"
        assert len(cache.root.children) == 2, "公共前缀没有被合并为共享边"
        shared = next(child for child in cache.root.children if child.key_tokens[0] == 0)
        assert shared.key_tokens == [0, 1, 2, 3], "共享边的 token 序列不正确"
        assert shared.terminal is True, "已登记的完整路径应标记为 terminal"
        assert len(shared.children) == 1 and shared.children[0].key_tokens == [4], "新增后缀没有挂到共享边下"
        assert shared.children[0].terminal is True, "新增完整路径的后缀节点应标记为 terminal"
        print("✅ 多路径前缀命中选择正确！")

        # 3. 前缀拆分验证
        hit_prefix, miss_suffix, hit_len = cache.split_prompt([0, 1, 2, 3, 4, 5])
        assert hit_len == 5, "Hit Length 计算错误！"
        assert hit_prefix == [0, 1, 2, 3, 4], "可复用前缀拆分错误！"
        assert miss_suffix == [5], "待重算后缀拆分错误！"

        hit_prefix2, miss_suffix2, hit_len2 = cache.split_prompt([7, 6, 5])
        assert hit_len2 == 0, "无命中时 Hit Length 应为 0！"
        assert hit_prefix2 == [], "无命中时前缀应为空！"
        assert miss_suffix2 == [7, 6, 5], "无命中时后缀应保持原样！"
        assert cache.split_prompt([0, 1, 2, 3, 4]) == ([0, 1, 2, 3, 4], [], 5), "完整命中时 suffix 应为空"
        assert cache.match_prefix([]) == 0, "空 prompt 不应产生命中"
        assert cache.split_prompt([]) == ([], [], 0), "空 prompt 的拆分结果不正确"
        try:
            cache.insert([])
        except ValueError:
            pass
        else:
            raise AssertionError('空 token 路径不能登记')
        # 反向分裂：新路径是旧路径的前缀，shared 节点仍需成为 terminal
        reverse_cache = SimpleRadixCache()
        reverse_cache.insert([4, 5, 6, 7])
        reverse_cache.insert([4, 5])
        assert reverse_cache.match_prefix([4, 5, 8]) == 2, "反向边分裂未保留 terminal"
        print("✅ 前缀拆分与回退逻辑正确！")

        print("\n 所有测试通过：共享边、最长命中和 prompt 拆分逻辑正确。")

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
        elif isinstance(e, AssertionError):
            print("代码可能未完成，导致了断言失败")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_radix_attention()
