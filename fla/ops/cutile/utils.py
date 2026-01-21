import torch


def next_power_of_2(n):
    return 1 if n <= 0 else 1 << (n-1).bit_length()
