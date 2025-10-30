import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsBasedLoss(nn.Module):
    """
    Physics-informed loss for wildfire spread prediction.

    Components:
      1. BCE loss (data fidelity)
      2. Spatial continuity penalty: fire should spread from nearby active zones
      3. Directional penalty (optional): fire spread aligns with wind & slope

    Args:
        lambda_spatial (float): weight for spatial continuity term (scales multiplicative factor)
        lambda_directional (float): weight for directional term (currently unused by default)
        neighborhood_size (int): radius of spatial neighborhood (in pixels)
        pos_weight (float | None): positive class weight for BCE (aligns with BCE setup elsewhere)
    """
    def __init__(
        self,
        lambda_spatial: float = 0.2,
        lambda_directional: float = 0.0,
        neighborhood_size: int = 2,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.lambda_spatial = lambda_spatial
        self.lambda_directional = lambda_directional
        self.neighborhood_size = neighborhood_size
        # Store pos_weight and compute BCE per-pixel via functional API in forward
        if pos_weight is not None:
            # single-class pos_weight tensor
            pw = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer("_pos_weight", pw, persistent=False)
        else:
            self.register_buffer("_pos_weight", torch.empty(0), persistent=False)

        # Precompute kernel for morphological dilation
        kernel = torch.ones(
            (1, 1, 2 * neighborhood_size + 1, 2 * neighborhood_size + 1),
            dtype=torch.float32
        )
        self.register_buffer("kernel", kernel)

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor | None = None,
        wind_dir: torch.Tensor | None = None,
        slope: torch.Tensor | None = None,
    ):
        """
        Args:
            y_hat (Tensor): predicted logits, shape (B, H, W) or (B, 1, H, W)
            y (Tensor): ground truth fire mask, shape (B, H, W) or (B, 1, H, W)
            x (Tensor, optional): input tensor with time dimension to extract previous-day binary AF mask.
                                  Expected shape (B, T, C, H, W), with last channel being binary AF.
            wind_dir (Tensor, optional): wind direction in degrees (B, 1, H, W)
            slope (Tensor, optional): slope (B, 1, H, W)
        """
        if y_hat.ndim == 3:
            y_hat = y_hat.unsqueeze(1)
        if y.ndim == 3:
            y = y.unsqueeze(1)

        # Per-pixel BCE
        pos_w = self._pos_weight if self._pos_weight.numel() > 0 else None
        per_pixel_bce = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=pos_w, reduction="none")  # (B,1,H,W)

        # Default multiplicative factor is 1 everywhere
        factor = torch.ones_like(per_pixel_bce)

        # If full input x is provided, derive previous-day AF binary mask and compute a spatial continuity factor
        if x is not None and x.ndim == 5:
            # previous day (last observation) and last channel (binary AF) -> shape (B,1,H,W)
            # x shape expected: (B, T, C, H, W)
            x_prev_bin = x[:, -1, -1, ...].unsqueeze(1)
            prev_fire = (x_prev_bin > 0.5).float()

            # Dilate neighborhood of previous fire
            dilated = F.conv2d(prev_fire, self.kernel, padding=self.neighborhood_size)
            dilated = torch.clamp(dilated, 0, 1)

            # Compute how much predicted fire lies outside plausible spread region
            pred_probs = torch.sigmoid(y_hat)
            # Use LeakyReLU to keep small negative gradients
            outside_mask = F.leaky_relu(pred_probs - dilated, negative_slope=0.01)

            # Multiplicative factor: 1 + lambda_spatial * outside_mask
            factor = factor + self.lambda_spatial * outside_mask

            # Directional term disabled by default unless wind/slope are explicitly provided in usable form
            # Note: wind_dir feature in dataset is transformed (sin of degrees), so a directional penalty
            # based on an absolute angle isn't directly applicable without extra preprocessing.

        # Final scalar loss: mean over pixels and batch of factor * per-pixel BCE
        total = (factor * per_pixel_bce).mean()
        return total

    @staticmethod
    def shift_tensor(tensor, dx, dy):
        """
        Approximate shift of tensor by dx, dy (wind vector) using bilinear sampling.
        dx, dy in normalized coordinates [-1, 1]
        """
        B, C, H, W = tensor.shape
        # Normalize dx, dy to roughly one pixel = 2/H or 2/W
        dx_norm = dx / (W / 2)
        dy_norm = dy / (H / 2)

        # Create grid
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=tensor.device),
            torch.linspace(-1, 1, W, device=tensor.device),
            indexing="ij"
        )
        x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
        y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)

        new_x = x_grid - dx_norm.squeeze(1)
        new_y = y_grid - dy_norm.squeeze(1)

        grid = torch.stack((new_x, new_y), dim=-1)
        shifted = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return shifted
