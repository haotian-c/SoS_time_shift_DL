# Structural Similarity Index (SSIM) for PyTorch

This directory provides a PyTorch implementation of the Structural Similarity Index (SSIM), which is integrated into loss function to encourage learning of structural information in speed-of-sound (SoS) images.

---

## Files

- `ssim.py`: Core implementation of SSIM using PyTorch and `torch.nn.functional`.

---

## Usage

### import

```python
from ssim import SSIM, ssim
```

---

## Dependencies

- PyTorch 

---

## Acknowledgments

This implementation is adapted from the open-source repository [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) by Po-Hsun Su.

---
