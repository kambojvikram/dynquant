from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class QuantizedPackedTensor:
    """A real (integer) quantized tensor stored in a packed bitstream.

    Supports per-channel quantization (axis=0) for Linear layers.
    """

    bits: int
    shape: Tuple[int, ...]
    packed: torch.Tensor  # uint8, 1-D, CPU
    scale: torch.Tensor   # float32, 1-D (size=1 for per-tensor, size=out_channels for per-channel)
    zero: torch.Tensor    # int32, 1-D
    is_constant: bool = False
    constant_value: float = 0.0


def _qmax(bits: int) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    return (1 << int(bits)) - 1


def _looks_like_embedding_param(name: str) -> bool:
    n = str(name).lower()
    # Common HF names: model.embed_tokens, embed_tokens, wte
    return any(x in n for x in ("embed_tokens", ".embed_tokens", "wte", ".wte", "embedding", ".embed"))


def quantize_embedding_symmetric_per_row_4bit(
    weight: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, float]:
    """Symmetric 4-bit quantization with per-row scales.

    Stores values in unsigned uint4 (0..15) by using a fixed zero-point of 8:
        q_u = clamp(round(w/scale) + 8, 0, 15)
        deq = (q_u - 8) * scale

    Scale is per-row (axis=0), i.e. one scale per token embedding vector.
    """

    if weight.ndim != 2:
        raise ValueError("Embedding symmetric per-row quant expects a 2D weight")

    w = weight.detach().float()

    # Constant tensor edge case
    if w.numel() > 0 and (w.max() - w.min()).abs() <= eps:
        q = torch.zeros_like(w, dtype=torch.uint8)
        return q, torch.tensor([0.0], device=w.device), torch.tensor([0], device=w.device), True, float(w.min().item())

    # Per-row max-abs
    max_abs = torch.amax(torch.abs(w), dim=1, keepdim=True)  # [rows, 1]
    # Use int4 signed range [-8, 7] (16 levels). Map to uint4 via +8.
    scale = max_abs / 7.0
    scale = torch.where(scale <= eps, torch.ones_like(scale), scale)

    q = torch.round(w / scale) + 8.0
    q = torch.clamp(q, 0.0, 15.0).to(dtype=torch.uint8)

    # Store per-row scale/zero as 1-D vectors [rows]
    scale_v = scale.reshape(-1).to(dtype=torch.float32)
    zero_v = torch.full((w.shape[0],), 8, dtype=torch.int32, device=w.device)
    return q, scale_v, zero_v, False, 0.0


def quantize_to_uint(
    weight: torch.Tensor,
    bits: int,
    *,
    eps: float = 1e-12,
    axis: int = 0,  # Default to per-channel (dim 0)
    group_size: int = 128  # Added group_size support
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, float]:
    """Quantize a float weight tensor to uint values in [0, 2^bits-1].
    
    Supports Group-wise quantization (group_size=128 by default).
    If group_size <= 0 or group_size >= In_Dim, falls back to per-channel.
    """

    if bits >= 16:
        raise ValueError("quantize_to_uint expects bits < 16")

    w = weight.detach()
    w_f = w.float()
    
    # Handle constant tensor edge case
    if w_f.numel() > 0 and (w_f.max() - w_f.min()).abs() <= eps:
         q = torch.zeros_like(w_f, dtype=torch.uint8)
         return q, torch.tensor([0.0]), torch.tensor([0]), True, float(w_f.min().item())

    # --- Group-Wise Reshaping Logic ---
    use_group = False
    original_shape = w_f.shape
    
    # Only support group quantization for standard Linear layers [Out, In]
    # where In (dim 1) is divisible by group_size (or we handle padding).
    if w_f.ndim == 2 and group_size > 0 and group_size < w_f.shape[1]:
        use_group = True
        out_features, in_features = w_f.shape
        
        # Calculate padding if needed
        if in_features % group_size != 0:
            pad_len = group_size - (in_features % group_size)
            w_f = torch.nn.functional.pad(w_f, (0, pad_len), mode="constant", value=0.0)
            in_features += pad_len
            
        # Reshape to [Out, Num_Groups, Group_Size]
        num_groups = in_features // group_size
        w_f = w_f.view(out_features, num_groups, group_size)
        
        # New "channel" is actually (Out, Group) -> flattened dim 0 and 1
        # To reuse the 'per-channel' logic below where we reduce over the LAST dim:
        # We want min/max over the 'Group_Size' dimension (-1).
        
        # Current w_f: [Out, Groups, GroupSize]
        # We process each [Out, Group] as an independent block.
    
    # Per-channel (or Per-Block) min/max
    # We always reduce over the last dimension(s) now.
    
    if use_group:
         # Reduce over GroupSize (dim=2)
         w_min = torch.amin(w_f, dim=-1, keepdim=True) # [Out, Groups, 1]
         w_max = torch.amax(w_f, dim=-1, keepdim=True) # [Out, Groups, 1]
    elif w_f.ndim > 1:
        reduce_dims = [d for d in range(w_f.ndim) if d != axis]
        w_min = torch.amin(w_f, dim=reduce_dims, keepdim=True)
        w_max = torch.amax(w_f, dim=reduce_dims, keepdim=True)
    else:
        w_min = torch.amin(w_f).view(1)
        w_max = torch.amax(w_f).view(1)

    qmax = _qmax(bits)
    
    # w_min/max are already broadcast-ready due to keepdim=True above (or view)
    w_min_v = w_min
    w_max_v = w_max

    # --- MSE OPTIMIZATION ---
    # Now operates per-group if enabled.
    
    best_score = torch.full_like(w_min_v, float("inf")).flatten() # [Out*Groups] or [Out]
    # We need to be careful with shapes here. 
    # If grouped: w_f is [Out, Groups, GroupSize], w_min is [Out, Groups, 1].
    # We want to sum errors over GroupSize (dim -1).
    
    best_scale = torch.zeros_like(w_min_v)
    best_zero = torch.zeros_like(w_min_v)

    if use_group:
        # Reshape score tracker to [Out, Groups]
        best_score = torch.full((w_f.shape[0], w_f.shape[1]), float("inf"), device=w_f.device)
    else:
        best_score = torch.full((w_f.shape[0],), float("inf"), device=w_f.device)

    search_range = [1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.85, 0.80]
    w_range = w_max_v - w_min_v
    
    for alpha in search_range:
        margin = (1.0 - alpha) * 0.5 * w_range
        curr_min = w_min_v + margin
        curr_max = w_max_v - margin
        
        scale_t = (curr_max - curr_min) / float(qmax)
        scale_t = torch.where(scale_t.abs() <= eps, torch.ones_like(scale_t), scale_t)
        zero_t = torch.round(-curr_min / scale_t)
        
        q_f = torch.round(w_f / scale_t) + zero_t
        q_f = torch.clamp(q_f, 0.0, float(qmax))
        w_recon = (q_f - zero_t) * scale_t
        
        err = (w_f - w_recon).pow(2)
        
        # Calculate MSE for the block/channel
        if use_group:
            mse = err.sum(dim=-1) # Sum over GroupSize -> [Out, Groups]
        elif w_f.ndim > 1:
            reduce_dims = [d for d in range(w_f.ndim) if d != axis]
            mse = err.sum(dim=reduce_dims) 
        else:
            mse = err

        improve_mask = mse < best_score
        
        # Update
        best_score = torch.where(improve_mask, mse, best_score)
        
        # Expand mask for broadcasting against Scale/Zero
        # If grouped: Mask [Out, Groups] -> [Out, Groups, 1]
        if use_group:
             mask_v = improve_mask.unsqueeze(-1)
        else:
             # Standard channel broadcasting
             shape_view = [1] * w_f.ndim
             shape_view[axis] = -1
             mask_v = improve_mask.view(shape_view)

        best_scale = torch.where(mask_v, scale_t, best_scale)
        best_zero = torch.where(mask_v, zero_t, best_zero)

    scale_t = best_scale
    zero_t = best_zero
    
    q_f = torch.round(w_f / scale_t) + zero_t
    q_f = torch.clamp(q_f, 0.0, float(qmax))
    q = q_f.to(dtype=torch.uint8)

    # --- Cleanup / Restoration ---
    if use_group:
        # Flatten groups back into original shape if we padded? 
        # Actually, for packed storage, we might want to keep the group structure 
        # OR flatten q back to [Out, In].
        # But scale/zero need to be stored as [Out, Groups].
        
        # Flatten Q back to [Out, In_Padded] -> Crop if padded -> [Out, In]
        q = q.view(original_shape[0], -1) 
        if q.shape[1] > original_shape[1]:
            q = q[:, :original_shape[1]]
            
        # Scale/Zero are distinct: they are [Out, Groups] flattened to 1D?
        # Standard GPTQ stores them as [Out * Groups, 1] or similar.
        # We will flatten them.
        return q, scale_t.flatten(), zero_t.flatten().to(dtype=torch.int32), False, 0.0

    return q, scale_t.flatten(), zero_t.flatten().to(dtype=torch.int32), False, 0.0
    # --- MSE OPTIMIZATION END ---

    # Original Code (Replaced by above):
    # scale_t = (w_max_v - w_min_v) / float(qmax)
    # ...


def pack_bits(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a 1-D uint tensor into a uint8 bitstream (little-endian).

    Fast paths:
    - 8-bit: no packing
    - 4-bit: 2 values per byte
    - 2-bit: 4 values per byte

    Slow fallback:
    - 3-bit: generic bitstream packing (crosses byte boundaries)
    """

    if q.dtype != torch.uint8:
        raise ValueError("q must be uint8")

    if q.ndim != 1:
        q = q.reshape(-1)

    bits = int(bits)
    if bits not in (2, 3, 4, 8):
        raise ValueError("Only 2/3/4/8-bit packing is supported")

    qmax = _qmax(bits)
    if int(q.max().cpu().item()) > qmax:
        raise ValueError("q contains values outside range for given bits")

    n = int(q.numel())
    if n == 0:
        return torch.empty((0,), dtype=torch.uint8)

    if bits == 8:
        return q.contiguous().cpu()

    # Fast path: 4-bit (2 values per byte)
    if bits == 4:
        q2 = q
        if (n % 2) != 0:
            pad = torch.zeros((1,), dtype=torch.uint8, device=q2.device)
            q2 = torch.cat([q2, pad], dim=0)
        q2 = q2.reshape(-1, 2)
        packed = (q2[:, 0] & 0x0F) | ((q2[:, 1] & 0x0F) << 4)
        return packed.contiguous().cpu()

    # Fast path: 2-bit (4 values per byte)
    if bits == 2:
        q4 = q
        rem = n % 4
        if rem != 0:
            pad = torch.zeros((4 - rem,), dtype=torch.uint8, device=q4.device)
            q4 = torch.cat([q4, pad], dim=0)
        q4 = q4.reshape(-1, 4)
        packed = (
            (q4[:, 0] & 0x03)
            | ((q4[:, 1] & 0x03) << 2)
            | ((q4[:, 2] & 0x03) << 4)
            | ((q4[:, 3] & 0x03) << 6)
        )
        return packed.contiguous().cpu()

    # 3-bit: Use vectorized packer (much faster than generic fallback)
    if bits == 3:
        # Re-enabling fast vectorized packer
        return _pack_bits_3bit_vectorized(q).cpu()



    # Slow fallback: generic bitstream packing.
    return _pack_bits_slow(q, bits)


def unpack_bits(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Unpack a packed uint8 bitstream into a 1-D uint8 tensor (little-endian)."""

    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8")

    bits = int(bits)
    if bits not in (2, 3, 4, 8):
        raise ValueError("Only 2/3/4/8-bit packing is supported")

    numel = int(numel)
    if numel <= 0:
        return torch.empty((0,), dtype=torch.uint8, device=packed.device)

    # For 3-bit, we use the vectorized unpacker
    if bits == 3:
        # SAFE MODE: Use slow unpacker
        # return _unpack_bits_slow_3bit(packed.cpu(), bits, numel).to(device=packed.device)
        return _unpack_bits_3bit_vectorized(packed, numel).to(device=packed.device)


    # For 2/4/8 bits, we can unpack on the device (CPU or GPU)
    if bits == 8:
        if packed.numel() < numel:
            raise ValueError("packed too short")
        return packed[:numel].contiguous()

    if bits == 4:
        # 1 byte -> 2 values (low nibble first)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        out = torch.empty((packed.numel() * 2,), dtype=torch.uint8, device=packed.device)
        out[0::2] = low
        out[1::2] = high
        return out[:numel].contiguous()

    if bits == 2:
        # 1 byte -> 4 values (2 bits each, LSB-first)
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        out = torch.empty((packed.numel() * 4,), dtype=torch.uint8, device=packed.device)
        out[0::4] = v0
        out[1::4] = v1
        out[2::4] = v2
        out[3::4] = v3
        return out[:numel].contiguous()

    # Should be unreachable
    return _unpack_bits_slow_3bit(packed.cpu(), bits, numel).to(device=packed.device)


def _pack_bits_slow(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Generic bit packer (LSB-first) used for awkward widths (e.g., 3-bit)."""

    if q.dtype != torch.uint8:
        raise ValueError("q must be uint8")
    q_cpu = q.contiguous().cpu().reshape(-1)
    n = int(q_cpu.numel())
    if n == 0:
        return torch.empty((0,), dtype=torch.uint8)

    total_bits = n * int(bits)
    out_len = (total_bits + 7) // 8
    out = torch.zeros((out_len,), dtype=torch.uint8)

    byte = 0
    bitpos = 0
    out_i = 0
    for v in q_cpu.tolist():
        v = int(v)
        remaining = int(bits)
        shift = 0
        while remaining > 0:
            avail = 8 - bitpos
            take = remaining if remaining <= avail else avail
            mask = (1 << take) - 1
            chunk = (v >> shift) & mask
            byte |= (chunk << bitpos)

            bitpos += take
            remaining -= take
            shift += take

            if bitpos == 8:
                out[out_i] = byte
                out_i += 1
                byte = 0
                bitpos = 0

    if bitpos != 0:
        out[out_i] = byte

    return out


def _unpack_bits_slow_3bit(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Generic bit unpacker (LSB-first) for awkward widths (3-bit)."""

    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8")

    bits = int(bits)
    numel = int(numel)
    if numel <= 0:
        return torch.empty((0,), dtype=torch.uint8)

    qmax = _qmax(bits)
    out = torch.empty((numel,), dtype=torch.uint8)

    byte_i = 0
    bitpos = 0
    cur = int(packed[byte_i].item()) if packed.numel() > 0 else 0
    for i in range(numel):
        v = 0
        filled = 0
        while filled < bits:
            avail = 8 - bitpos
            take = (bits - filled) if (bits - filled) <= avail else avail
            mask = (1 << take) - 1
            chunk = (cur >> bitpos) & mask
            v |= (chunk << filled)

            bitpos += take
            filled += take
            if bitpos == 8:
                byte_i += 1
                if byte_i < packed.numel():
                    cur = int(packed[byte_i].item())
                bitpos = 0

        out[i] = int(v & qmax)

    return out


def _pack_bits_3bit_vectorized(q: torch.Tensor) -> torch.Tensor:
    """Vectorized 3-bit packing (LSB-first). Packs 8 values into 3 bytes."""
    # q: [N], uint8
    n = q.numel()
    pad_len = (8 - (n % 8)) % 8
    if pad_len > 0:
        q = torch.cat([q, torch.zeros(pad_len, dtype=torch.uint8, device=q.device)])
    
    q = q.view(-1, 8).int()
    
    # Extract 8 values
    v0, v1, v2, v3, v4, v5, v6, v7 = q.unbind(dim=1)
    
    # Byte 0: v0(3) | v1(3) | v2(2)
    b0 = (v0 & 0x07) | ((v1 & 0x07) << 3) | ((v2 & 0x03) << 6)
    
    # Byte 1: v2(1) | v3(3) | v4(3) | v5(1)
    b1 = ((v2 >> 2) & 0x01) | ((v3 & 0x07) << 1) | ((v4 & 0x07) << 4) | ((v5 & 0x01) << 7)
    
    # Byte 2: v5(2) | v6(3) | v7(3)
    b2 = ((v5 >> 1) & 0x03) | ((v6 & 0x07) << 2) | ((v7 & 0x07) << 5)
    
    packed = torch.stack([b0, b1, b2], dim=1).flatten().to(torch.uint8)
    
    # Trim padding
    total_bits = n * 3
    out_len = (total_bits + 7) // 8
    return packed[:out_len]


def _unpack_bits_3bit_vectorized(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Vectorized 3-bit unpacking (LSB-first). Unpacks 3 bytes into 8 values."""
    # packed: [M], uint8
    
    # Ensure we have enough bytes for chunks of 3
    n_packed = packed.numel()
    pad_len = (3 - (n_packed % 3)) % 3
    if pad_len > 0:
        packed = torch.cat([packed, torch.zeros(pad_len, dtype=torch.uint8, device=packed.device)])
    
    # Reshape to [N, 3]
    packed = packed.view(-1, 3).int()
    b0, b1, b2 = packed.unbind(dim=1)
    
    # v0: b0 & 0x07
    v0 = b0 & 0x07
    
    # v1: (b0 >> 3) & 0x07
    v1 = (b0 >> 3) & 0x07
    
    # v2: ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
    v2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
    
    # v3: (b1 >> 1) & 0x07
    v3 = (b1 >> 1) & 0x07
    
    # v4: (b1 >> 4) & 0x07
    v4 = (b1 >> 4) & 0x07
    
    # v5: ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
    v5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
    
    # v6: (b2 >> 2) & 0x07
    v6 = (b2 >> 2) & 0x07
    
    # v7: (b2 >> 5) & 0x07
    v7 = (b2 >> 5) & 0x07
    
    # Stack and flatten
    unpacked = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).flatten().to(torch.uint8)
    
    return unpacked[:numel].contiguous()


def quantize_packed(weight: torch.Tensor, bits: int, *, param_name: str | None = None) -> QuantizedPackedTensor:
    """Quantize and pack a weight tensor into a compressed representation."""

    bits = int(bits)
    if bits >= 16:
        raise ValueError("quantize_packed expects bits < 16")

    # Optimization: Ensure quantization happens on GPU if available, even if model is on CPU.
    # The MSE search is computationally expensive on CPU.
    compute_device = weight.device
    if compute_device.type == "cpu" and torch.cuda.is_available():
        compute_device = torch.device("cuda")
        weight = weight.to(compute_device)

    # Special case: embeddings in symmetric 4-bit with per-row scales.
    # Rationale: embeddings are sensitive to zero-point bias; per-row scales fit token-row statistics well.
    if bits == 4 and param_name and _looks_like_embedding_param(param_name) and weight.ndim == 2:
        q, scale, zero, is_const, const_val = quantize_embedding_symmetric_per_row_4bit(weight)
    else:
        # Default path:
        # - Per-channel quantization (axis=0) for 2D weights
        # - Group-wise quantization (group_size=128) for better low-bit accuracy
        q, scale, zero, is_const, const_val = quantize_to_uint(weight, bits, axis=0, group_size=128)
    
    # Important for performance: pack on the source device (GPU/CPU) and only
    # transfer the packed byte-stream back to CPU for storage.
    packed = pack_bits(q.reshape(-1), bits)
    
    # If we moved to GPU for compute, move results back to CPU for storage/return
    # to avoid OOM during the loop.
    return QuantizedPackedTensor(
        bits=bits,
        shape=tuple(int(d) for d in weight.shape),
        packed=packed.contiguous().cpu(),
        scale=scale.cpu(),
        zero=zero.cpu(),
        is_constant=bool(is_const),
        constant_value=float(const_val),
    )


def dequantize_packed(qt: QuantizedPackedTensor, *, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize a packed representation back to a float tensor."""

    numel = 1
    for d in qt.shape:
        numel *= int(d)
    if qt.is_constant:
        # Respect the device of the packed tensor if possible, or default to CPU
        dev = qt.packed.device
        return torch.full(qt.shape, float(qt.constant_value), dtype=dtype, device=dev)

    # Ensure unpack_bits runs on the same device as qt.packed
    q = unpack_bits(qt.packed, qt.bits, numel).to(dtype=torch.float32)
    q = q.reshape(qt.shape)
    # Handle per-channel or per-group broadcasting
    scale = qt.scale.to(dtype=torch.float32)
    zero = qt.zero.to(dtype=torch.float32)

    # Detect if we are using group quantization
    # Normal channel-wise: scale len = Out
    # Group-wise: scale len = Out * (In / GroupSize)
    if scale.numel() > qt.shape[0]:
        # Assume Group-Wise Layout
        out_features, in_features = qt.shape
        num_groups = scale.numel() // out_features
        
        # Robust group size deduction
        if in_features % num_groups == 0:
            group_size = in_features // num_groups
        else:
            # Padding was likely used. Assume 128 as standard, or infer.
            # If 128 works with ceil division, use it.
            if (in_features + 127) // 128 == num_groups:
                group_size = 128
            else:
                # Fallback to integer division (risky if padded)
                group_size = in_features // num_groups

        # Reshape scales/zeros to [Out, Groups, 1] for broadcasting
        scale = scale.view(out_features, num_groups, 1)
        zero = zero.view(out_features, num_groups, 1)
        
        # Handle Padding for Q reshape
        # If the input was padded during quant, q was cropped before storage.
        # We need to re-pad it to view it as [Out, Groups, GroupSize].
        pad_len = 0
        if (in_features % group_size) != 0:
             pad_len = group_size - (in_features % group_size)
             q = torch.nn.functional.pad(q, (0, pad_len), value=0)

        # Reshape q to [Out, Groups, GroupSize]
        q_grouped = q.view(out_features, num_groups, group_size)
        
        deq = (q_grouped - zero) * scale
        
        # Remove padding and restore shape
        deq = deq.reshape(out_features, -1)
        if pad_len > 0:
            deq = deq[:, :in_features]
            
        return deq.reshape(qt.shape).to(dtype=dtype)
    
    # Fallback to Channel-wise logiczero = qt.zero.to(dtype=torch.float32)
    
    if scale.numel() > 1:
        # Per-channel: assume axis 0
        view_shape = [1] * len(qt.shape)
        view_shape[0] = -1
        scale = scale.view(view_shape)
        zero = zero.view(view_shape)
    
    deq = (q - zero) * scale
    return deq.to(dtype=dtype)


def serialize_quantized_packed(qt: QuantizedPackedTensor) -> Dict[str, torch.Tensor]:
    """Convert to a tensor dict suitable for safetensors saving."""

    return {
        "packed": qt.packed.to(dtype=torch.uint8, device="cpu"),
        "scale": qt.scale.to(dtype=torch.float32, device="cpu"),
        "zero": qt.zero.to(dtype=torch.int32, device="cpu"),
        "bits": torch.tensor([qt.bits], dtype=torch.int16),
        "shape": torch.tensor(list(qt.shape), dtype=torch.int32),
        "is_constant": torch.tensor([1 if qt.is_constant else 0], dtype=torch.uint8),
        "constant_value": torch.tensor([qt.constant_value], dtype=torch.float32),
    }


def quantize_layer(layer: nn.Module, bits: int) -> Optional[QuantizedPackedTensor]:
    """Quantize a module's `weight` into a packed integer representation.

    Returns `None` for bits>=16 (kept as fp16) or if the module has no weight.
    """

    bits = int(bits)
    if bits >= 16:
        return None
    if not hasattr(layer, "weight"):
        return None
    w = getattr(layer, "weight")
    if not isinstance(w, torch.Tensor):
        return None
    return quantize_packed(w, bits)


def maybe_quantize_module(module: nn.Module, bits: Optional[int]) -> Optional[QuantizedPackedTensor]:
    if bits is None:
        return None
    return quantize_layer(module, int(bits))
