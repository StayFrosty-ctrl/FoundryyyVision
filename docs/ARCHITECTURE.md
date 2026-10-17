# Architecture & Open-Source Reference Landscape

## 1. Open-Source Foundations Scouted

FoundryyyVision synthesizes patterns from state-of-the-art open-source sports computer vision projects:

| Domain | Reference Repositories | Key Capabilities Adopted |
| :--- | :--- | :--- |
| **Baseball CV** | [`BaseballCV/BaseballCV`](https://github.com/BaseballCV/BaseballCV), [`2205DUP/baseballtracker`](https://github.com/2205DUP/baseballtracker), [`jbarger/Baseball-Tracker`](https://github.com/jbarger/Baseball-Tracker) | Specialized detectors for tiny high-velocity baseballs (`ball_tracking`), `bat_tracking`, `glove_tracking`, `homeplate_tracking`, and `pitcher_hitter_catcher` roles; 3D trajectory physics and swing biomechanics. |
| **American Football CV** | [`wannahappyaroundme/football-tracking`](https://github.com/wannahappyaroundme/football-tracking), [`TheoViel/nfl_impact_detection`](https://github.com/TheoViel/nfl_impact_detection), Kaggle NFL Helmet Assignment | Synchronized Sideline/Endzone/All-22 player & helmet tracking, team jersey/helmet clustering, line-of-scrimmage estimation, pre-snap formation recognition, run/pass classification, and helmet impact detection. |
| **Multi-Object Tracking (MOT)** | [`mikel-brostrom/boxmot`](https://github.com/mikel-brostrom/boxmot), [`MCG-NJU/SportsMOT`](https://github.com/MCG-NJU/SportsMOT) | Pluggable motion + appearance trackers (BoT-SORT, Deep OC-SORT, ByteTrack) with Camera Motion Compensation (CMC) for panning/tilting broadcast and sideline cameras. |
| **Field Calibration & Homography** | [`roboflow/sports`](https://github.com/roboflow/sports), [`SoccerNet/sn-calibration`](https://github.com/SoccerNet/sn-calibration) | Keypoint-based planar homography mapping 2D pixel bounding boxes and footpoints onto standardized real-world coordinates (NFL 120x53.3 yd grid & MLB 90 ft diamond). |
| **Temporal Action Spotting** | [`SoccerNet/osl-actionspotting`](https://github.com/SoccerNet/osl-actionspotting) | Frame-accurate spotting of discrete play boundaries (`pitch_release`, `bat_contact`, `ball_snap`, `pass_forward`, `tackle`, `whistle`). |

---

## 2. Core Pipeline Design

Every video frame flows through a deterministic, low-latency pipeline (`foundryvision.core.pipeline.VisionPipeline`):

1. **Ingest (`core.ingest`)**: Decodes RTSP/WebRTC streams or local video files into timestamped `FramePacket` objects.
2. **Perception (`core.perception`)**: Runs object detection (players, ball, bat, glove, helmet, referee), pose estimation (skeletal keypoints), and field landmark detection.
3. **Calibration (`core.calibration`)**: Computes a planar homography matrix $H$ from field keypoints (yard lines/hash marks or bases/home plate) to project image footpoints $(u, v)$ into real-world field coordinates $(X, Y)$.
4. **Tracking (`core.tracking`)**: Associates detections across frames into persistent `Tracklet` trajectories with velocity vectors and camera motion compensation.
5. **Identity (`core.identity`)**: Resolves persistent athlete identity using team color clustering, jersey number OCR, and appearance Re-ID embeddings, smoothed over track history.
6. **Sport Plugin (`sports.baseball` / `sports.football`)**: Applies sport-specific state machines, event spotting, and scouting analytics.
