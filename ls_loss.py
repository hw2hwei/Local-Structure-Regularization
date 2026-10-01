"""Local Structure Regularization losses for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LocalStructureLossBase(nn.Module):
    """Base loss that expects raw logits and applies softmax internally."""

    def __init__(
        self,
        num_classes,
        ignore_index=255,
        label_smooth_kernel=3,
        boundary_kernel=3,
        eps=1e-7,
    ):
        super().__init__()
        if isinstance(ignore_index, int):
            ignore_index = (ignore_index,)
        elif isinstance(ignore_index, (list, tuple, set)):
            ignore_index = tuple(ignore_index)
        else:
            raise TypeError("ignore_index must be an int or a collection of ints")

        self.ignore_index = ignore_index
        self.num_classes = int(num_classes)
        self.label_smooth_kernel = self._odd_kernel(label_smooth_kernel)
        self.boundary_kernel = self._odd_kernel(boundary_kernel)
        self.eps = float(eps)

        average_kernel = torch.ones(1, 1, self.label_smooth_kernel, self.label_smooth_kernel)
        self.register_buffer("average_kernel", average_kernel / average_kernel.numel())
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            ).view(1, 1, 3, 3),
        )

    @staticmethod
    def _odd_kernel(size):
        size = max(1, int(size))
        return size if size % 2 else size + 1

    def _depthwise_conv(self, tensor, kernel):
        channels = tensor.shape[1]
        kernel = kernel.to(device=tensor.device, dtype=tensor.dtype)
        kernel = kernel.expand(channels, 1, -1, -1)
        return F.conv2d(tensor, kernel, padding=kernel.shape[-1] // 2, groups=channels)

    def _valid_mask(self, target):
        valid = torch.ones_like(target, dtype=torch.bool)
        for ignored_label in self.ignore_index:
            valid &= target != ignored_label
        return valid

    def _check_inputs(self, logits, target):
        if logits.ndim != 4:
            raise ValueError(f"logits must have shape [B, C, H, W], got {tuple(logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"target must have shape [B, H, W], got {tuple(target.shape)}")
        if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
            raise ValueError("logits and target batch/spatial dimensions must match")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits have {logits.shape[1]} classes; expected {self.num_classes}"
            )
        if logits.device != target.device:
            raise ValueError("logits and target must be on the same device")

        valid = self._valid_mask(target)
        invalid = valid & ((target < 0) | (target >= self.num_classes))
        if invalid.any():
            bad_labels = torch.unique(target[invalid]).detach().cpu().tolist()
            raise ValueError(
                f"target contains labels {bad_labels} outside [0, {self.num_classes})"
            )
        return valid

    def _target_fields(self, target, valid):
        safe_target = target.masked_fill(~valid, 0).long()
        one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=self.sobel_x.dtype)
        one_hot = one_hot * valid.unsqueeze(1).to(dtype=one_hot.dtype)

        radius = self.boundary_kernel // 2
        padded = F.pad(one_hot, (radius, radius, radius, radius), mode="replicate")
        local_max = F.max_pool2d(padded, self.boundary_kernel, stride=1)
        local_min = -F.max_pool2d(-padded, self.boundary_kernel, stride=1)
        boundary = (local_max - local_min > 0).any(dim=1, keepdim=True)
        boundary = boundary & valid.unsqueeze(1)

        if self.label_smooth_kernel > 1:
            smoothed = self._depthwise_conv(one_hot, self.average_kernel)
        else:
            smoothed = one_hot
        target_orientation, target_magnitude = self._orientation_and_magnitude(smoothed)
        return boundary, target_orientation, target_magnitude

    def _orientation_and_magnitude(self, field):
        grad_x = self._depthwise_conv(field, self.sobel_x)
        grad_y = self._depthwise_conv(field, self.sobel_y)

        j_xx = (grad_x * grad_x).sum(dim=1, keepdim=True)
        j_yy = (grad_y * grad_y).sum(dim=1, keepdim=True)
        j_xy = (grad_x * grad_y).sum(dim=1, keepdim=True)

        orientation_x = j_xx - j_yy
        orientation_y = 2.0 * j_xy
        magnitude = torch.sqrt(orientation_x.square() + orientation_y.square() + self.eps)
        orientation = torch.cat((orientation_x, orientation_y), dim=1)
        orientation = orientation / (magnitude + self.eps)
        return orientation, magnitude

    def _components(self, logits, target):
        valid = self._check_inputs(logits, target)
        if not valid.any():
            zero = logits.sum() * 0.0
            return zero, zero

        boundary, target_orientation, target_magnitude = self._target_fields(target, valid)
        if not boundary.any():
            zero = logits.sum() * 0.0
            return zero, zero

        probabilities = F.softmax(logits, dim=1)
        predicted_orientation, predicted_magnitude = self._orientation_and_magnitude(probabilities)

        target_max = target_magnitude.flatten(2).amax(dim=2).unsqueeze(-1).unsqueeze(-1)
        target_magnitude = target_magnitude / (target_max + self.eps)
        weight = boundary.to(dtype=logits.dtype) * target_magnitude.detach()
        denominator = weight.sum()
        if denominator.detach().item() <= 0:
            zero = logits.sum() * 0.0
            return zero, zero

        orientation_dot = (predicted_orientation * target_orientation).sum(dim=1, keepdim=True)
        orientation_dot = orientation_dot.clamp(-1.0, 1.0)
        orientation_loss = ((1.0 - orientation_dot) * weight).sum() / (denominator + self.eps)

        predicted_max = predicted_magnitude.flatten(2).amax(dim=2).unsqueeze(-1).unsqueeze(-1)
        predicted_magnitude = predicted_magnitude / (predicted_max + self.eps)
        magnitude_loss = self._magnitude_loss(
            predicted_magnitude, target_magnitude.detach(), weight
        )
        return orientation_loss, magnitude_loss

    def _magnitude_loss(self, predicted, target, weight):
        raise NotImplementedError


class LocalStructureOrientationLoss(_LocalStructureLossBase):
    """Match local boundary orientation in the predicted class-probability field."""

    def _magnitude_loss(self, predicted, target, weight):
        return predicted.sum() * 0.0

    def forward(self, logits, target):
        orientation_loss, _ = self._components(logits, target)
        return orientation_loss


class LocalStructureMagnitudeLoss(_LocalStructureLossBase):
    """Match normalized local boundary-transition magnitude."""

    def __init__(self, *args, mag_loss_type="l1", **kwargs):
        super().__init__(*args, **kwargs)
        if mag_loss_type not in {"l1", "mse", "log_l1"}:
            raise ValueError("mag_loss_type must be 'l1', 'mse', or 'log_l1'")
        self.mag_loss_type = mag_loss_type

    def _magnitude_loss(self, predicted, target, weight):
        if self.mag_loss_type == "l1":
            difference = (predicted - target).abs()
        elif self.mag_loss_type == "mse":
            difference = (predicted - target).square()
        else:
            difference = (torch.log(predicted + self.eps) - torch.log(target + self.eps)).abs()
        return (difference * weight).sum() / (weight.sum() + self.eps)

    def forward(self, logits, target):
        _, magnitude_loss = self._components(logits, target)
        return magnitude_loss


class LocalStructureRegularizationLoss(_LocalStructureLossBase):
    """Combine local structure orientation (LSO) and magnitude (LSM) losses."""

    def __init__(
        self,
        *args,
        lambda_orientation=1.0,
        lambda_magnitude=1.0,
        mag_loss_type="l1",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mag_loss_type not in {"l1", "mse", "log_l1"}:
            raise ValueError("mag_loss_type must be 'l1', 'mse', or 'log_l1'")
        self.lambda_orientation = float(lambda_orientation)
        self.lambda_magnitude = float(lambda_magnitude)
        self.mag_loss_type = mag_loss_type

    def _magnitude_loss(self, predicted, target, weight):
        if self.mag_loss_type == "l1":
            difference = (predicted - target).abs()
        elif self.mag_loss_type == "mse":
            difference = (predicted - target).square()
        else:
            difference = (torch.log(predicted + self.eps) - torch.log(target + self.eps)).abs()
        return (difference * weight).sum() / (weight.sum() + self.eps)

    def forward(self, logits, target):
        orientation_loss, magnitude_loss = self._components(logits, target)
        return (
            self.lambda_orientation * orientation_loss
            + self.lambda_magnitude * magnitude_loss
        )