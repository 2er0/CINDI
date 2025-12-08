# KnewImp
This is an official implementation of KnewImp approach, which has been accepted as the poster of NeurIPS' 2024 conference (main track).


An Arxiv preprint can be found in https://arxiv.org/abs/2406.15762

# Source

https://github.com/JustusvLiebig/NewImp

# Changes

- Updated to be compatible with the latest versions of PyTorch and other dependencies.
  - `functorch` is now a part of PyTorch, so the separate installation is no longer required.
  - `functorch` replaced with `torch.func` in the code.
- Fixed float precision miss matches