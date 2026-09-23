import torch

from comb2.op_utils import rank


def test_rank_float16_long_cpu_axis_uses_integer_positions():
    values = torch.zeros((4096, 2), dtype=torch.float16)
    expected = torch.arange(1, 4097, dtype=torch.int64).to(torch.float16)

    actual = rank(values, dim=0)

    torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 1], expected, rtol=0, atol=0)
