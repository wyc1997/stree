from contextlib import contextmanager
import torch


@contextmanager
def cuda_time(desc, l=None, layer=None):
    if layer is not None and layer != 1:
        yield
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1_000_000)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        if l is not None:
            l.append(start.elapsed_time(end))
        else:
            print("{} layer{} run time:".format(desc, layer), start.elapsed_time(end))

