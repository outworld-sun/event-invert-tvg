"""PTS-based video sampling shared by actors, frozen judges, and evaluation."""

import bisect
import math
from collections import OrderedDict
from pathlib import Path

from vtg.utils.io import digest
from vtg.video.budget import frame_pixel_budget, resize_shape, visual_tokens

DECODER_VERSION = "pyav_pts_dynamic_v2"


def scan_video(path):
    import av

    timestamps = []
    indices = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None:
                timestamps.append(float(frame.pts * frame.time_base))
                indices.append(index)
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if duration is None and container.duration:
            duration = float(container.duration / av.time_base)
    if not timestamps or duration is None or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Missing valid PTS or duration: {path}")
    if timestamps != sorted(timestamps):
        raise ValueError(f"Nonmonotonic decoded PTS: {path}")
    return {
        "duration_sec": duration,
        "pts": timestamps,
        "decode_indices": indices,
        "decoder_version": DECODER_VERSION,
    }


def select_pts(pts, interval, cfg):
    """Uniformly sample the entire request, repeating short clips if needed."""
    start, end = interval
    if not all(math.isfinite(v) for v in interval) or not 0 <= start < end:
        raise ValueError("Invalid requested interval")
    first = bisect.bisect_left(pts, start)
    last = bisect.bisect_left(pts, end)
    if first == last:
        raise ValueError("No frames inside interval")
    count = max(cfg["min_frames"], min(cfg["max_frames"], round((end - start) * cfg["fps"])))
    count = min(cfg["max_frames"], math.ceil(count / 2) * 2)
    chosen = []
    for index in range(count):
        target = start + (index + 0.5) * (end - start) / count
        position = bisect.bisect_left(pts, target, first, last)
        candidates = [i for i in (position - 1, position) if first <= i < last]
        chosen.append(min(candidates, key=lambda i: (abs(pts[i] - target), i)))
    return {
        "indices": chosen,
        "pts": [pts[i] for i in chosen],
        "repeated_frames": count - len(set(chosen)),
        "effective_fps": count / (end - start),
    }


class VideoAdapter:
    """Keep one decoded request and a bounded metadata cache per worker."""

    def __init__(self, root, cfg, processor_config=None):
        self.root = Path(root)
        self.cfg = cfg
        self.processor_config = processor_config or {}
        self.cache = {}
        self.scans = OrderedDict()

    def load(self, request):
        import av
        import numpy as np
        import torch
        from PIL import Image

        path = self.root / request["video_relpath"]
        stat = path.stat()
        source = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if source not in self.scans:
            self.scans[source] = scan_video(path)
            if len(self.scans) > self.cfg.get("scan_cache_size", 32):
                self.scans.popitem(last=False)
        self.scans.move_to_end(source)
        scan = self.scans[source]
        choice = select_pts(scan["pts"], request["interval"], self.cfg)
        key = digest([source, request, DECODER_VERSION, self.cfg, self.processor_config, choice])
        if key in self.cache:
            return self.cache[key]
        decode_indices = [scan["decode_indices"][i] for i in choice["indices"]]
        wanted = set(decode_indices)
        decoded = {}
        budget = frame_pixel_budget(len(decode_indices), self.cfg)
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index not in wanted:
                    continue
                height, width = resize_shape(
                    frame.height, frame.width, self.cfg["min_pixels_per_frame"], budget
                )
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                array = np.asarray(image.resize((width, height), Image.Resampling.BICUBIC)).copy()
                decoded[index] = torch.from_numpy(array).permute(2, 0, 1)
                if len(decoded) == len(wanted):
                    break
        if len(decoded) != len(wanted):
            raise ValueError("Video changed or decoding failed after PTS scan")
        frames = torch.stack([decoded[i] for i in decode_indices])
        tokens = visual_tokens(frames.shape[0], frames.shape[2], frames.shape[3])
        if tokens > self.cfg["visual_token_cap"]:
            raise ValueError(f"Actual visual tokens {tokens} exceed configured hard cap")
        meta = {
            **choice,
            "video_sample_hash": key,
            "visual_tokens": tokens,
            "shape": list(frames.shape),
            "frame_pixel_budget": budget,
            "decoder_version": DECODER_VERSION,
        }
        result = frames, meta
        self.cache = {key: result}
        return result
