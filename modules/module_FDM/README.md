## Usage

We provide an example of using APM (Amplitude-Phase Masker) and ACPA (Adaptive Channel Phase Attention), which are integrated into the model to operate on intermediate features.

```python
import torch
from .freq_masker import MaskModule
from .phase_attn import PhaseAttention

bsz, c, h, w = 3, 2048, 13, 13
immediate_feature = torch.rand(bsz, c, h, w)

# APM-S
apm_s = MaskModule([1,1,h,w])
# APM-M
apm_m = MaskModule([1,c,h,w])
# ACPA
acpa = PhaseAttention(c)

enhanced_feature = apm_s(immediate_feature) # apm_m(immedidate_feature)
final_feature = acpa(enhanced_feature)
```