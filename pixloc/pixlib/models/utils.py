import torch


def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)


def checkpointed(cls, do=True):
    '''Adapted from the DISK implementation of Michał Tyszkiewicz.'''
    assert issubclass(cls, torch.nn.Module)

    # cls:torch.nn.Sequential
    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            
            """
            torch.nn.Sequential().forward
            for module in self:
                input = module(input)
            """            
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                """
                通常の設定でforwardを実行すると、中間層における演算結果は、勾配の計算で再利用するため、
                すべて保持するようになっています5。演算結果が容易に得られる層に関しては、演算結果を保持
                しないようにすることで計算グラフの消費メモリを減るので、バッチサイズを大きくすることがで
                きます（forwardを再計算するよりも、バッチサイズを大きくしたほうが高速になる可能性が高い）。
                """
                return torch.utils.checkpoint.checkpoint(super_fwd, *args, **kwargs)
            else:
                """
                通常の設定でforwardを実行
                """
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls
