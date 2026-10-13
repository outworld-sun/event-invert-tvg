"""Invert4TVG pixel allocation, with Qwen's 28-pixel spatial alignment."""

import math

SPATIAL_FACTOR = 28
TEMPORAL_FACTOR = 2


def frame_pixel_budget(frame_count, config):
    """Return the per-frame area budget, not a fixed square image size."""
    if frame_count < 2 or frame_count % TEMPORAL_FACTOR:
        raise ValueError("Frame count must be positive and divisible by two")
    maximum = config["max_pixels_per_frame"]
    if config["allocation"] == "dynamic":
        maximum = max(
            min(maximum, config["total_pixels"] * TEMPORAL_FACTOR / frame_count),
            int(config["min_pixels_per_frame"] * 1.05),
        )
    return maximum


def resize_shape(height, width, minimum, maximum):
    """Match the upstream round/floor/ceil policy and preserve aspect ratio.

    The minimum and maximum are target areas. Alignment can move the final
    area outside the nominal minimum, just as in the original smart_resize.
    A separate visual-token check enforces the actual hard memory budget.
    """
    factor = SPATIAL_FACTOR
    if min(height, width) < factor or max(height, width) / min(height, width) > 200:
        raise ValueError("Video dimensions are too small or have an extreme aspect ratio")
    new_height = round(height / factor) * factor
    new_width = round(width / factor) * factor
    area = new_height * new_width
    if area > maximum:
        scale = math.sqrt(height * width / maximum)
        new_height = max(factor, math.floor(height / scale / factor) * factor)
        new_width = max(factor, math.floor(width / scale / factor) * factor)
    elif area < minimum:
        scale = math.sqrt(minimum / (height * width))
        new_height = math.ceil(height * scale / factor) * factor
        new_width = math.ceil(width * scale / factor) * factor
    return new_height, new_width


def visual_tokens(frames, height, width):
    return frames // TEMPORAL_FACTOR * (height // SPATIAL_FACTOR) * (width // SPATIAL_FACTOR)
