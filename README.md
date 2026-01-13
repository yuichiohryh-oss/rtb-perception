# rtb_perception

Diff-based spawn detection stabilized by tracking state.

## Install

```bash
pip install -e .[dev]
```

## CLI

```bash
python -m rtb_perception.run_tracker --video path/to/video.mp4 --out out_dir --debug
```

Outputs:
- `out_dir/events.jsonl`
- `out_dir/debug/frame_000123.jpg` when `--debug` is set

## JSONL schema

Each line is one event (UTF-8 JSONL).

```json
{"event":"spawn","frame":12,"t":0.4,"track_id":1,"bbox":[10,20,30,40],"source":"diff","iou":0.78,"age":1,"missed":0}
{"event":"update","frame":13,"t":0.43,"track_id":1,"bbox":[12,22,32,42],"source":"diff","iou":0.66,"age":2,"missed":0}
{"event":"disappear","frame":30,"t":1.0,"track_id":1,"bbox":[12,22,32,42],"source":"diff","iou":null,"age":19,"missed":6}
```

Required keys:
- `event`: `spawn` | `update` | `disappear`
- `frame`: 0-based frame index
- `t`: seconds or null
- `track_id`: integer ID
- `bbox`: `[x1, y1, x2, y2]`
- `source`: always `diff`

Optional keys:
- `iou`, `age`, `missed`, `meta`

## Notes

The tracker uses IoU >= 0.3, greedy matching, and spawn confirmation with two consecutive frames.
