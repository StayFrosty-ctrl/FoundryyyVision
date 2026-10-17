# Project Context & Agent Handoff

Read this first if you are an agent (or human) picking up FoundryyyVision. It captures the decisions and discussion that led to the current codebase.

## Vision

A real-time video intelligence platform that tracks and identifies individuals and recognizes their actions/activities, specialized for **scouting** in two sports:

- **Baseball**
- **American football** (not soccer)

Both are discrete, play-by-play sports (pitch-by-pitch, snap-to-whistle), which makes play segmentation and clip indexing a shared concern.

## Key decisions

1. **One platform, two plugins.** A sport-agnostic core (ingest → perception → calibration → tracking → identity → events) plus per-sport plugins (`sports/baseball`, `sports/football`) implementing `SportPlugin` in `src/foundryvision/sports/base.py`.
2. **Scout open source first, then define.** The scouting results are in `docs/ARCHITECTURE.md`. Repos worth reading in depth:
   - Baseball: `BaseballCV/BaseballCV` (ball/bat/glove/home-plate/role detectors)
   - American football: `wannahappyaroundme/football-tracking`, `TheoViel/nfl_impact_detection`, Kaggle NFL Helmet Assignment data
   - Tracking: `mikel-brostrom/boxmot`, `MCG-NJU/SportsMOT`
   - Calibration: `roboflow/sports`, `SoccerNet/sn-calibration`
   - Action spotting: `SoccerNet/osl-actionspotting`
3. **Capture assumptions matter more than model choice.** The camera setup (tripod/phone, fixed elevated, All-22, or broadcast) still needs to be decided. It determines what's feasible.
4. **Prove one hard capability end to end before broadening.** Candidates: baseball pitch/ball tracking, or football snap/play segmentation.

## Intended competitive edge (hypotheses, not validated)

- Output scouting language (formations, routes, pitch tendencies), not just boxes
- Near-real-time feedback for in-game adjustments, not next-day film breakdown
- No special hardware: works from ordinary video through automatic field homography
- Persistent athlete identity across plays, games, and seasons
- One engine serving both the baseball and football scouting markets

These claims need validation against existing products (Hudl, Catapult, Hawk-Eye, Rapsodo, TrackMan, and others) before being treated as fact.

## Current state of the code (honest assessment)

**Working, with 5 passing tests:**
- Data schemas, pipeline orchestration, ring buffer, event log
- Pure-Python planar homography (DLT; no numpy dependency at runtime)
- Greedy IoU and center-distance tracker; temporal jersey/team identity voting

**Placeholders to replace with real models and data:**
- **No real perception yet.** `PassthroughPerception` only consumes pre-filled detections. Next step: wire up YOLO/RF-DETR (BaseballCV weights for baseball).
- **Synthetic data only.** `cli.py` generates fake frame sequences. No real video ingest (OpenCV/RTSP) yet.
- **Baseball velocities are not physically meaningful.** `pitch_mph = |vy| * 3.4` and the exit-velocity constant are arbitrary pixel-to-mph scales. Real speed estimates need calibrated 3D or plane-aware measurement, plus a high-fps camera.
- **Football logic is heuristic.** Formation classification, line-of-scrimmage estimation, snap detection, and run/pass rules are simple thresholds. They are placeholders for learned models.
- The tracker is a simplified stand-in. Plan to swap in BoxMOT (BoT-SORT/ByteTrack with camera motion compensation).
- `api/` has serializers only. There is no FastAPI app yet.

## Running

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m foundryvision.cli --sport football
PYTHONPATH=src python3 -m foundryvision.cli --sport baseball
```

## Open questions for the user

1. Which sport is the first vertical slice?
2. Which camera setup is the target?
3. Who is the first customer: high school/college coaches, pro scouts, or individual players?
4. Is there access to real labeled footage?

## Suggested next steps

1. Answer the open questions above.
2. Add a real video ingest path (OpenCV file/RTSP reader implementing `FrameSource`).
3. Integrate a real detector (BaseballCV weights or YOLO) behind `PerceptionModel`.
4. Swap the tracker for BoxMOT.
5. Build a small labeled evaluation set and define metrics (detection mAP, IDF1/HOTA, event F1, latency).
