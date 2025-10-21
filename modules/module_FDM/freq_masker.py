import torch
import torch.nn as nn


class MaskModule(nn.Module):
    """Lightweight Amplitude-Phase Masker (APM) with lazy parameter init."""

    def __init__(self, shape=None):
        # APM-S shape: [1, 1, h, w]
        # APM-M shape: [1, c, h, w]
        super().__init__()
        self.target_shape = tuple(shape) if shape is not None else None
        self.register_parameter("mask_amplitude", None)
        self.register_parameter("mask_phase", None)

    def _init_from_shape(self, shape, device):
        amplitude = torch.ones(shape, dtype=torch.float32, device=device)
        phase = torch.ones(shape, dtype=torch.float32, device=device)
        self.register_parameter("mask_amplitude", nn.Parameter(amplitude))
        self.register_parameter("mask_phase", nn.Parameter(phase))

    def _ensure_initialized(self, x):
        expected_shape = self.target_shape or x.shape
        if self.mask_amplitude is None or self.mask_amplitude.shape != expected_shape:
            self._init_from_shape(expected_shape, x.device)

    def reset_masks(self):
        if self.mask_amplitude is None or self.mask_phase is None:
            return
        with torch.no_grad():
            self.mask_amplitude.fill_(1.0)
            self.mask_phase.fill_(1.0)

    def forward(self, x):
        x = x.float()
        self._ensure_initialized(x)

        freq_domain = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        amplitude = torch.abs(freq_domain)
        phase = torch.angle(freq_domain)

        mask_amplitude = torch.sigmoid(self.mask_amplitude)
        mask_phase = torch.sigmoid(self.mask_phase)

        adjusted_amplitude = mask_amplitude * amplitude
        adjusted_phase = mask_phase * phase

        adjusted_freq = torch.polar(adjusted_amplitude, adjusted_phase)
        adjusted_x = torch.fft.ifft2(torch.fft.ifftshift(adjusted_freq, dim=(-2, -1)), dim=(-2, -1)).real

        return adjusted_x

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        amplitude_key = prefix + "mask_amplitude"
        phase_key = prefix + "mask_phase"

        if amplitude_key in state_dict:
            tensor = state_dict[amplitude_key]
            self.target_shape = tensor.shape
            if self.mask_amplitude is None or self.mask_amplitude.shape != tensor.shape:
                self.mask_amplitude = nn.Parameter(tensor.clone().detach())
            else:
                self.mask_amplitude.data.copy_(tensor)
        elif strict and self.mask_amplitude is not None:
            missing_keys.append(amplitude_key)

        if phase_key in state_dict:
            tensor = state_dict[phase_key]
            self.target_shape = tensor.shape
            if self.mask_phase is None or self.mask_phase.shape != tensor.shape:
                self.mask_phase = nn.Parameter(tensor.clone().detach())
            else:
                self.mask_phase.data.copy_(tensor)
        elif strict and self.mask_phase is not None:
            missing_keys.append(phase_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
