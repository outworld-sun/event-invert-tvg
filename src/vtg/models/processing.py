"""Apply the same chat template and video geometry in every execution mode."""


def load_processor(config):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config["paths"]["model_path"])
    processor.tokenizer.padding_side = "left"
    processor.image_processor.min_pixels = config["video"]["min_pixels_per_frame"]
    processor.image_processor.max_pixels = config["video"]["max_pixels_per_frame"]
    return processor


def encode_video(processor, prompt, frames, meta, config, copies=1):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video", "video": "visual_input"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # The decoder has already applied the frame-dependent pixel allocation.
    # A second processor resize would invalidate both the budget and input hash.
    batch = processor(
        text=[text] * copies,
        videos=[frames] * copies,
        fps=[meta["effective_fps"]] * copies,
        do_resize=False,
        return_tensors="pt",
        padding=True,
    )
    if batch["input_ids"].shape[1] > config["training"]["max_prompt_tokens"]:
        raise ValueError("Prompt exceeds the configured token budget (never truncate video tokens)")
    tokens = batch["video_grid_thw"].prod(-1) // 4
    if any(int(value) > config["video"]["visual_token_cap"] for value in tokens):
        raise ValueError("Processor output exceeds the actual visual token budget")
    if any(int(value) != meta["visual_tokens"] for value in tokens):
        raise ValueError("Processor unexpectedly changed decoded video geometry")
    return batch
