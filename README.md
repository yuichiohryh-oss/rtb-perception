# rtb_perception

Diff-based spawn detection stabilized by tracking state.

## Install

```bash
pip install -e .[dev]
```

## CLI

```bash
python -m rtb_perception.run_tracker --video path/to/video.mp4 --out out_dir --debug --iou-thresh 0.3 --confirm-frames 2 --max-missed 5 --diff-threshold 25 --kernel-size 3 --min-area 80
```

Hand UI mis-detections can be reduced by limiting diff to the board ROI:

```bash
python -m rtb_perception.run_tracker --video path/to/video.mp4 --out out_dir --debug --roi-top 0.14 --roi-bottom 0.74
```

For compressed videos (e.g. YouTube), use blur and a larger diff step:

```bash
python -m rtb_perception.run_tracker --video path/to/video.mp4 --out out_dir --debug --roi-top 0.16 --roi-bottom 0.68 --blur 5 --diff-step 2
```

Parameters:
- `--diff-threshold`: pixel intensity threshold for diff mask.
- `--blur`: Gaussian blur kernel size for diff (0 disables; even values are rounded up).
- `--diff-step`: frame step for diff (1 compares to previous frame).
- `--kernel-size`: morphology kernel size for opening/closing.
- `--min-area`: minimum bbox area to keep.
- `--roi-top`: top ratio of ROI.
- `--roi-bottom`: bottom ratio of ROI.
- `--roi-left`: left ratio of ROI.
- `--roi-right`: right ratio of ROI.
- `--side-split`: board height ratio for enemy/friendly split.
- `--kind-window`: frames to accumulate movement for kind_guess.
- `--kind-move-thresh`: movement threshold for area_spell vs unit.

Examples:
- `--side-split 0.50` で敵/味方推定
- `--kind-window 6 --kind-move-thresh 10` で範囲スペル推定

Outputs:
- `out_dir/events.jsonl`
- `out_dir/debug/frame_000123.jpg` when `--debug` is set

## JSONL schema

Each line is one event (UTF-8 JSONL).

```json
{"event":"spawn","frame":12,"t":0.4,"track_id":1,"bbox":[10,20,30,40],"source":"diff","iou":0.78,"age":1,"missed":0}
{"event":"update","frame":13,"t":0.43,"track_id":1,"bbox":[12,22,32,42],"source":"diff","iou":0.66,"age":2,"missed":0}
{"event":"disappear","frame":30,"t":1.0,"track_id":1,"bbox":[12,22,32,42],"source":"diff","age":19,"missed":6}
```

Required keys:
- `event`: `spawn` | `update` | `disappear`
- `frame`: 0-based frame index
- `t`: seconds or null
- `track_id`: integer ID
- `bbox`: `[x1, y1, x2, y2]`
- `source`: always `diff`

Optional keys:
- `iou`, `age`, `missed`, `center`, `side`, `kind_guess`, `meta` (only included when available)

## Debug legend

- Thick green boxes: active tracks with `id`, `age`, `missed`
- Thin gray boxes: spawn candidates
- Thin yellow boxes: diff bboxes
- Thin light gray box: ROI used for diff

## Notes

The tracker uses IoU >= 0.3, greedy matching, and spawn confirmation with two consecutive frames.
If you see spurious spawns from the hand UI, tune ROI ratios to focus on the board area.
