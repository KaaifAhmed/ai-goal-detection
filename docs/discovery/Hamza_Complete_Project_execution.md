# AI Goal Detector for Informal Football
## Research and Implementation Plan — Single Mobile Camera Behind the Goal

**Project objective:** Use video from one mobile camera placed behind the goalkeeper to decide whether a football is a **GOAL**, **NO GOAL**, or **UNCERTAIN**. The system must distinguish a ball that fully crosses the goal plane from a ball that is blocked by the goalkeeper, hits a goal marker/brick, goes wide, or passes over the goal.

**Primary priority:** Build and evaluate the Python computer-vision pipeline and dataset first. Do not build the backend or mobile application until the model and decision logic demonstrate reliable results in real footage.

---

## 1. Define the goal correctly

For a physical goal with a left marker and a right marker, define an imaginary **goal plane** between them. A goal is not simply “the ball appeared between the markers in one frame.”

A robust decision should consider:

1. The ball's position relative to the goal plane.
2. Whether the **entire ball** crossed the plane, not just its centre.
3. Whether the ball went between the markers and below the chosen crossbar/height boundary.
4. Whether the goalkeeper or a marker blocked or deflected it before crossing.
5. Whether the ball was visible clearly enough to make a reliable decision.

### Output classes

- **GOAL:** There is sufficient visual/geometric evidence that the whole ball crossed the goal plane inside the goal boundaries.
- **NO GOAL:** There is sufficient evidence that it did not cross correctly (blocked, wide, above the goal, or hit a marker and stayed out).
- **UNCERTAIN:** The ball is occluded, blurred, outside the frame, or too close to a boundary for reliable judgement.

Do not force every event into GOAL or NO GOAL. An honest UNCERTAIN result is safer than a confident guess.

> **Important limitation:** With one camera, some 3D situations are fundamentally ambiguous—especially when the ball is completely hidden at the exact crossing moment. The system must be allowed to return UNCERTAIN.

---

## 2. Fixed project boundaries

The first version assumes:

- One mobile camera only.
- Camera is behind the goalkeeper/goal, looking toward the playing area.
- Only ordinary 2D video frames are available; no depth sensor and no second camera.
- Camera remains stationary during a play.
- The goal is made from two physical markers (such as bricks); there may be no crossbar or net.
- Players, the goalkeeper, feet, shadows, and other objects can enter the image.
- The first milestone is offline Python processing of recorded clips.
- Backend, server deployment, and mobile application are later milestones.

### Suggested recording starting point

These are **initial experimental settings, not guaranteed requirements**:

| Setting | Initial recommendation | Why |
|---|---|---|
| Frame rate | 60 FPS if available; otherwise 30 FPS | More temporal samples for fast shots |
| Resolution | 1080p if practical | Helps preserve small ball details |
| Camera position | Fixed, behind and as close to centred on the goal as practical | More stable geometry |
| Camera height | Measure and record it; test practical heights around 1.2–2.0 m | Height affects perspective |
| Camera distance | Measure it; begin with a few metres behind the goal and test alternatives | Affects ball pixel size and view |
| Orientation | Avoid extreme sideways angles; record any offset | Helps geometric calibration |
| Stabilisation | Prefer a firmly mounted phone; test stabilisation rather than assuming it must be off | Digital stabilisation can change geometry |
| Focus/exposure | Lock where the phone permits, while avoiding very dark footage | Reduces frame-to-frame changes |
| Clip format | Original video with timestamps retained | Needed for motion and reproducible labels |

A fixed camera is a major advantage. If it moves, recalibrate before using the footage for metric geometry.

---

## 3. Variables and constraints to measure

Record measurements per **session/setup**, not repeatedly for every frame unless the value changes.

### A. Physical goal and ball

| Variable | Symbol / unit | How to measure | Why it matters |
|---|---|---|---|
| Ball diameter | `ball_diameter_m` (metres) | Measure the actual ball with a tape/ruler; record ball type | Helps estimate apparent-size scale and ball boundary |
| Ball type | `ball_type` | User-entered category, e.g. size 4, size 5, futsal | Different balls have different sizes |
| Left marker position | `left_marker_xyz_m` or ground coordinates | Measure relative to a chosen field origin | Defines left boundary |
| Right marker position | `right_marker_xyz_m` or ground coordinates | Measure relative to the same origin | Defines right boundary and goal width |
| Goal width | `goal_width_m` | Distance between the inner edges of the two markers | Defines allowed horizontal crossing area |
| Marker dimensions | width/height/depth in metres | Measure each brick/block | Helps distinguish marker contact and define boundaries |
| Goal height / upper boundary | `goal_height_m` | If a crossbar exists, measure it; if no crossbar exists, agree on a virtual height rule | Defines the upper goal boundary |
| Ground plane | `ground_plane` | Identify several points on the same flat playing surface | Needed for ground-plane mapping |
| Goal-line / goal-plane location | `goal_plane` | Mark the intended plane between the goal markers | Crossing test reference |
| Surface condition | `surface_type` | Dirt, grass, concrete, etc. | Ball bounce and rolling behaviour vary |

**Important:** If the informal goal has two bricks but no crossbar, there is no naturally defined upper boundary. The team must agree on a rule (for example, a defined virtual height) before training or evaluating a goal detector. Do not silently treat the top of the video or an arbitrary pixel row as the crossbar.

For formal goal-line logic, the whole ball must cross the plane. If the ball is modelled as a sphere with radius `r = ball_diameter_m / 2`, its centre must pass sufficiently beyond the plane for the trailing edge to cross. A centre crossing alone is not enough.

### B. Camera and video

| Variable | Symbol / unit | How to measure | Why it matters |
|---|---|---|---|
| Image width/height | `frame_width_px`, `frame_height_px` | Read from video metadata | Pixel coordinates depend on resolution |
| FPS and timestamps | `fps`, `timestamp_s` | Read actual video timestamps where possible | Converts frame displacement to motion |
| Camera intrinsics | `fx`, `fy`, `cx`, `cy` | Calibrate using a checkerboard/known calibration target, where possible | Maps 3D rays to image pixels |
| Lens distortion | distortion coefficients | Estimate during camera calibration | Wide-angle phone lenses distort geometry |
| Camera height | `camera_height_m` | Measure lens centre above ground | Helps determine geometry |
| Camera position/orientation | pose relative to goal | Measure or estimate from known points | Connects image to goal coordinates |
| Camera-to-goal distance | `camera_goal_distance_m` | Measure from lens centre to defined goal plane | Useful for calibration and reproducibility |
| Camera lateral offset | `camera_offset_m` | Measure offset from goal centreline | Corrects off-centre perspective |
| Camera pitch/roll/yaw | degrees | Measure/estimate from calibration | Tilt changes image geometry |
| Stabilisation/zoom state | metadata/config | Record phone settings | Changing zoom or digital crop invalidates calibration |

Do not assume that “the goal occupies 50% of the image” or that a ball has a fixed pixel diameter at a particular distance. These are camera-dependent heuristics, not physical laws.

### C. Per-frame ball observations

For every frame in which the ball is labelled, record:

- `ball_visible`: visible / partially occluded / fully occluded / out of frame.
- `ball_bbox`: bounding box `(x_min, y_min, x_max, y_max)` in pixels.
- `ball_center_px`: `(u, v)`.
- `ball_diameter_px` or width/height of the bounding box, if measurable.
- `ball_detection_quality`: clear / blurry / partial / uncertain.
- `ball_track_id`: identity through the clip.
- `occluder`: goalkeeper / other player / marker / unknown / none.
- `event_phase`: approaching / near-plane / crossing / leaving / stopped / unknown.

Use the same pixel coordinate convention everywhere: origin at top-left, `u` increases to the right, and `v` increases downward.

### D. Per-clip event labels

Give every clip a clip-level ground-truth label:

- `goal`
- `no_goal_blocked_by_goalkeeper`
- `no_goal_marker_contact`
- `no_goal_wide`
- `no_goal_above_boundary`
- `no_goal_stopped_before_plane`
- `uncertain_occluded`
- `uncertain_blurry_or_out_of_frame`
- `other`

Also record `contact_event` (none / goalkeeper / marker / ground / other), whether the ball was fully visible at the decisive moment, and how the label was verified.

---

## 4. Recommended computer-vision approach

Use a **hybrid system**, not one end-to-end classifier as the only decision-maker.

### Stage 1 — Detect objects

Fine-tune a small object detector to detect at least:

- `football`
- `goalkeeper`
- `person` (optional separate class if goalkeeper needs distinguishing)
- `left_marker`
- `right_marker`

Add a `referee` or other class only if it helps the use case. Do not label every foot or shadow as a football. The ball detector should be trained on hard negatives such as shoes, white stones, plastic bags, bright patches, and other ball-like objects.

A pretrained detector may already detect sports balls, but performance on a small, blurry football in your specific behind-goal footage must be measured before deciding whether fine-tuning is necessary.

### Stage 2 — Track the ball over time

Use detections from consecutive frames to maintain one ball track. A Kalman filter can predict short gaps, but a prediction is **not proof** that the ball was seen or crossed the line. Keep a flag showing whether a position is observed or merely predicted.

For an initial prototype:
- Simple nearest-neighbour association or a Kalman filter may be enough.
- Use ByteTrack or another multi-object tracker if needed.
- Do not trust a tracker to recover the ball through long occlusions without visual evidence.

### Stage 3 — Calibrate the scene and define the virtual goal

At setup, ask the user to identify the two markers and calibrate the camera/ground plane. Use known ground points and camera calibration to map pixels to ground-plane coordinates for a ball that is rolling on the ground.

For a single camera:
- Ground-plane homography can estimate positions of points **on the ground plane**.
- It does not directly recover the true 3D location of an airborne ball.
- Apparent ball size can provide a rough depth cue, but is noisy.
- Physics-constrained trajectory fitting can help with airborne segments if the camera is calibrated and enough correctly timed frames are visible. It is not guaranteed to produce a unique answer in every scene.

### Stage 4 — Estimate crossing and apply rule logic

Use geometry and temporal evidence to determine whether the ball crossed the defined goal plane. Include ball radius, not just the centre point.

The decision engine should check:
1. Is this the same ball track over the relevant frames?
2. Is there enough observed evidence near the goal plane?
3. Did the ball move from the playing side to the goal side?
4. Did the entire ball pass the plane within the lateral boundaries and below the agreed upper boundary?
5. Is there evidence that the goalkeeper or a marker stopped/deflected it before crossing?
6. Is the evidence strong enough, or should the result be `UNCERTAIN`?

Do not decide “goal” just because the ball is between the two markers in a single frame. Perspective can make a ball in front of the goal appear to be inside it.

---

## 5. Physics-constrained fitting: when and how

For a visible airborne segment, a basic model is:

`P(t) = P0 + V0 * t + 0.5 * G * t²`

where:
- `P0 = (X0, Y0, Z0)` is the starting 3D position.
- `V0 = (Vx, Vy, Vz)` is initial velocity.
- `G` is the gravity vector in the world coordinate system.
- The camera model projects `P(t)` to predicted pixel coordinates.

Use `scipy.optimize.least_squares` to adjust the six initial position/velocity parameters so predicted ball centres match observed pixel centres across several frames. Optionally add a low-weight apparent-diameter residual.

Constraints and cautions:
- Use real timestamps, not just frame numbers, if frame timing varies.
- Calibrate the camera before trusting metric 3D coordinates.
- Use robust loss such as `soft_l1` to reduce the effect of outlier detections.
- Split the trajectory at collisions/bounces/deflections because velocity changes suddenly.
- Exclude or down-weight fully occluded frames rather than inventing their pixel centres.
- A fitted trajectory is a hypothesis; compare it with observed frames and report uncertainty.
- Do not use this as the only method for ground-rolling balls; ground-plane geometry is usually simpler.

---

## 6. Dataset collection plan

Collect data from the **actual mobile-camera setup**. Public datasets can help with pretraining, but cannot replace local footage because your background, ball size in pixels, lighting, camera height, dirt surface, and brick goal are distinctive.

### A. Record controlled sessions

For each session, record a setup sheet containing:
- Session ID and date.
- Phone make/model if available.
- Resolution, FPS, zoom, stabilisation, focus/exposure settings.
- Camera height, distance behind goal plane, lateral offset, and angle.
- Goal width, marker dimensions, upper-boundary rule, ball diameter/type.
- Lighting/weather/surface.
- Any camera movement or recalibration.

### B. Record a balanced set of events

Include both positive and negative examples:

| Scenario | Examples to record | Ground-truth label |
|---|---|---|
| Clear goal | Ball fully crosses between markers | `goal` |
| Goalkeeper save | Ball hits keeper and stays out | `no_goal_blocked_by_goalkeeper` |
| Keeper deflection into goal | Ball hits keeper, then crosses completely | `goal` if it meets the rule |
| Marker hit | Ball hits left/right brick and rebounds out | `no_goal_marker_contact` |
| Marker hit then in | Ball hits marker then subsequently fully crosses | Apply crossing rule; often `goal` if it crosses correctly |
| Wide shot | Ball passes outside either marker | `no_goal_wide` |
| Above boundary | Ball passes above the agreed upper limit | `no_goal_above_boundary` |
| Stops short | Ball stops before crossing | `no_goal_stopped_before_plane` |
| Rolling slowly | Ball rolls across or stops near the plane | Goal/no-goal according to full crossing |
| Airborne shot | Ball travels through the air near the goal | Label based on verified outcome |
| Occlusion | Keeper/player hides the ball near the plane | `uncertain_occluded` if outcome cannot be verified |
| False objects | Shoes, stones, plastic, bright patches, shadows | No ball / hard negative |
| Lighting | Daylight, shade, evening, artificial lights | Correct event label |
| Camera variation | Centre and modest off-centre angles; multiple heights/distances | Correct event label per clip |
| Ball variation | Different real ball types/sizes | Correct event label per clip |

Record failed attempts, not just clean goals. Hard negative examples are crucial for avoiding false goals.

### C. Vary one factor at a time initially

Start with a stable baseline setup. Then vary camera distance, height, angle, lighting, ball type, and motion speed in controlled sessions. Record each configuration. This makes it possible to understand which changes hurt accuracy.

### D. Ground truth

For development, use an independent side-view camera or a human referee/review process to label whether the whole ball crossed. An additional camera may be used **only to establish training/evaluation labels**; the deployed system can still use one camera. If labels are determined from the same ambiguous camera view, errors in the labels can teach the model the wrong rule.

When annotators disagree, flag the clip for review rather than forcing a label.

### E. How much data?

There is no universal frame count that guarantees accuracy. Start with:
- A small pilot of about 20–50 clips covering every major event type.
- A first annotation batch of hundreds of representative frames for the ball detector.
- Expand to thousands of labelled frames and many independent clips as you find failure cases.

The number of **independent clips and varied scenarios** matters more than extracting thousands of near-identical consecutive frames from one shot.

---

## 7. Annotation format

Use CVAT or Roboflow for bounding-box annotations. Save a clip-level CSV/JSON manifest for the event labels and calibration data.

### Object-detection annotation

For each annotated frame, draw a tight box around the visible football. Label partly visible balls as partial/occluded if the tool supports attributes. Do not draw a football box around a shoe, shadow, or stone.

Suggested class IDs:

| ID | Class |
|---:|---|
| 0 | `football` |
| 1 | `goalkeeper` |
| 2 | `person` |
| 3 | `left_marker` |
| 4 | `right_marker` |

For a YOLO detector, export the matching YOLO detection format from CVAT/Roboflow. Make sure the dataset configuration uses the same class order.

### Example clip manifest

```csv
clip_id,session_id,event_label,contact_event,lighting,camera_setup_id,ball_type,verified
clip_0001,session_01,goal,none,daylight,setup_A,size_5,yes
clip_0002,session_01,no_goal_blocked_by_goalkeeper,goalkeeper,daylight,setup_A,size_5,yes
clip_0003,session_02,uncertain_occluded,unknown,evening,setup_B,futsal,review
```

### Example per-frame observation record

```json
{
  "clip_id": "clip_0001",
  "frame_index": 42,
  "timestamp_s": 1.4,
  "ball_visible": "partial",
  "ball_bbox_xyxy": [612, 341, 640, 369],
  "ball_center_px": [626, 355],
  "occluder": "goalkeeper",
  "event_phase": "near_plane"
}
```

Coordinates above are examples only. Use actual values from your footage.

### Calibration manifest

Store the measured setup separately, for example:

```json
{
  "setup_id": "setup_A",
  "frame_width_px": 1920,
  "frame_height_px": 1080,
  "fps": 60,
  "camera_height_m": 1.6,
  "camera_goal_distance_m": 3.5,
  "camera_offset_m": 0.0,
  "goal_width_m": 2.0,
  "ball_diameter_m": 0.22,
  "upper_boundary_defined": false
}
```

All values are **illustrative only**, not measured from your image. Replace them with actual measurements. The example deliberately marks the upper boundary as undefined because a two-brick goal without a crossbar needs an agreed rule.

---

## 8. Suggested Python technology stack

Use a small, testable stack first.

| Need | Recommended library/tool | Purpose |
|---|---|---|
| Video and image processing | `opencv-python` | Read frames, draw detections, calibration, homography, replay clips |
| Numerical calculations | `numpy` | Coordinates, vectors, geometry |
| Data tables/metrics | `pandas` | Dataset manifests, summaries, evaluation |
| Object detection | `ultralytics` (YOLO family) | Baseline pretrained detector and fine-tuning |
| Model training | `torch` | PyTorch training/inference ecosystem used by Ultralytics |
| Annotation | CVAT or Roboflow | Bounding boxes and dataset export |
| Trajectory optimisation | `scipy` | `least_squares` for physics-constrained fitting |
| Tracking | Start with OpenCV/Kalman; consider `supervision` or ByteTrack | Maintain ball track across frames |
| Plots and diagnostics | `matplotlib` | Trajectories, confidence, error analysis |
| Metrics | `scikit-learn` | Confusion matrix, precision/recall and classification metrics |
| Tests | `pytest` | Test crossing logic, calibration and edge cases |
| Optional experiment tracking | MLflow or simple CSV logs | Record model versions and experiments |

### Suggested starting environment

```bash
python -m venv .venv
```

Activate the environment, then install a baseline set:

```bash
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy pandas scipy scikit-learn matplotlib pytest
```

For GPU training, install a PyTorch build compatible with the chosen CUDA environment according to the official PyTorch installation selector. Google Colab or Kaggle can be used for initial GPU training. You do not need a GPU to start collecting data, annotating, calibrating, or testing the geometry.

Pin working package versions in `requirements.txt` after the baseline runs successfully.

---

## 9. Fine-tuning plan

### Step 1 — Establish a baseline before training

Run a pretrained small YOLO model on your own sample clips. Measure:
- How often the ball is detected when it is visible.
- How often a shoe, stone, shadow, or brick is incorrectly detected as the ball.
- Detection performance for tiny/blurred balls and partial occlusion.
- Processing speed on your likely target device.

This tells you whether fine-tuning is needed and which examples are missing.

### Step 2 — Fine-tune the ball detector

Train a small model such as a nano-sized YOLO variant using annotated local frames. Use a pretrained checkpoint rather than training from scratch.

Start with the `football` class as the highest priority. Add goalkeeper and marker classes if they help event logic. Keep a hard-negative set containing shoes, stones, shadows, plastic and other confusing objects.

Possible augmentations:
- Moderate brightness/contrast changes.
- Motion blur.
- Scale variation.
- Mild perspective and image transformations that reflect actual phone use.

Avoid augmentations that make the ball physically implausible or change the meaning of goal boundaries. Always inspect augmented samples.

### Step 3 — Keep event decisions separate at first

Do not initially train one black-box model to output GOAL/NO GOAL from isolated frames. A single frame often cannot show whether the ball crossed the plane or was blocked.

Instead:
1. Detector finds the ball in each frame.
2. Tracker links observations over time.
3. Geometry estimates where the ball is relative to the goal.
4. A deterministic decision engine applies the crossing rule.
5. The system returns GOAL, NO GOAL, or UNCERTAIN.

Later, if you have enough labelled clips, train a temporal event classifier as a second opinion. It should use multiple frames or track features, not a single still image alone.

### Step 4 — Prevent hallucinated goals

- Never count a tracker-predicted position as an observed ball detection.
- Require evidence of movement across the goal plane.
- Treat a goalkeeper occlusion at the critical moment as uncertain unless the later visible path resolves the outcome.
- Check whether the ball rebounds or stops before the plane.
- Reject detections that jump implausibly between frames.
- Use confidence thresholds tuned on validation clips, not chosen only by intuition.
- If the ball leaves the frame at the decisive moment, do not invent its position.
- Keep hard negative examples and review every false GOAL.

---

## 10. Evaluation: prove it works before building the app

Split data by **whole recording session or clip**, never randomly by individual frames. Adjacent frames are nearly duplicates; placing them in both train and test produces misleadingly high scores.

Maintain:
- Training set: used to fine-tune the detector.
- Validation set: used to choose thresholds and settings.
- Test set: held back until final evaluation.

Evaluate separately for clear goals, goalkeeper saves, marker impacts, wide shots, occlusions, lighting, and camera setups.

### Key metrics

| Metric | Why it matters |
|---|---|
| Ball detection precision | How many predicted balls were real balls |
| Ball detection recall | How many visible balls were found |
| False GOAL rate | Critical: how often a non-goal is incorrectly called a goal |
| Goal recall | How many real goals are recognised |
| UNCERTAIN rate | How often the system cannot decide |
| Decision accuracy by scenario | Shows where the system fails |
| Time to decision | Useful for future live use |
| Calibration / crossing error | Measures geometric position error where ground truth is available |

For a sports decision tool, false GOAL calls can be especially damaging. Optimise the threshold for low false-goal rate while reporting how many cases become UNCERTAIN. Do not report only overall accuracy.

Create a failure-review folder and add every missed ball, false ball, wrong crossing, and uncertain event to the next annotation round.

---

## 11. Recommended project folder structure

```text
ai_goal_detector/
├── data/
│   ├── raw_videos/          # Original videos; do not overwrite
│   ├── extracted_frames/
│   ├── annotations/
│   ├── manifests/
│   ├── calibration/
│   └── splits/              # Train/validation/test clip IDs
├── models/
│   ├── pretrained/
│   └── checkpoints/
├── src/
│   ├── video_io.py          # Read video and timestamps
│   ├── detect_ball.py       # YOLO inference
│   ├── track_ball.py        # Track IDs and short gaps
│   ├── calibrate_camera.py  # Intrinsics and scene calibration
│   ├── ground_plane.py      # Ground-plane mapping
│   ├── trajectory_fit.py    # Optional airborne trajectory fitting
│   ├── goal_geometry.py     # Virtual goal plane and ball radius
│   ├── decision_engine.py   # GOAL / NO GOAL / UNCERTAIN
│   └── visualize.py         # Draw tracks, boundaries and evidence
├── tests/
│   ├── test_goal_geometry.py
│   ├── test_decision_engine.py
│   └── test_edge_cases.py
├── runs/                    # Predictions and evaluation output
├── requirements.txt
└── README.md
```

---

## 12. Edge cases and expected behaviour

| Situation | Correct system behaviour |
|---|---|
| Ball crosses completely between markers | GOAL |
| Goalkeeper blocks the ball and it stays outside | NO GOAL |
| Goalkeeper touches it, but it then fully crosses | GOAL if the defined rule is satisfied |
| Ball hits a marker and rebounds out | NO GOAL |
| Ball hits a marker then crosses fully inside | Apply the agreed crossing rule; do not automatically reject it because contact occurred |
| Ball passes wide | NO GOAL |
| Ball goes above the defined upper boundary | NO GOAL |
| Ball centre crosses but part of ball remains outside | Not yet a goal |
| Player/foot enters the goal area | Do not treat as ball; object detector and temporal track must agree |
| Ball is hidden by goalkeeper at the exact moment | UNCERTAIN unless other evidence resolves it |
| Ball detection disappears for a few frames | Track may predict briefly, but do not use prediction alone as proof of crossing |
| Ball is tiny or blurred | Lower confidence; UNCERTAIN if evidence is insufficient |
| Camera shifts | Pause decisions and recalibrate |
| A shoe, stone, shadow or plastic resembles a ball | Reject based on appearance, tracking and scene consistency |
| Ball is stopped just in front of the goal plane | NO GOAL |
| Ball leaves the image before the result is visible | UNCERTAIN unless its crossing can be established reliably |

---

## 13. Sensor-based approach: optional future experiment

Do not start with sensors unless the camera-only baseline proves insufficient. Sensors can provide additional evidence, but a sensor triggered by any object might also trigger for a goalkeeper's foot or another object.

A sensor setup would need to detect a valid ball crossing rather than just movement. It also adds hardware cost, alignment, power and weather/dirt reliability concerns. First quantify the camera-only failure modes; then decide whether a sensor can address them.

---

## 14. Local-market validation in Karachi

Test with informal football groups on actual dirt/grass grounds, different phones, and common lighting conditions. Ask players to compare the system's decision and replay with a human-reviewed result.

The most useful early product is not a claim of perfect automatic refereeing. It is a transparent tool that:
- shows a replay with the ball track and virtual goal boundaries;
- explains GOAL / NO GOAL / UNCERTAIN;
- avoids false certainty on occluded close calls;
- works on the phones and grounds players actually use.

Record supported phone types, lighting conditions, setup limits, false-goal rate, and uncertainty rate. Do not market accuracy claims until they have been measured on held-out local footage.

---

## 15. Implementation milestones

1. **Define rules and measure the setup.** Agree on goal width, marker dimensions, upper boundary, ball-size handling, and camera position.
2. **Collect a pilot dataset.** Capture all event categories, including hard negatives and goalkeeper occlusions.
3. **Annotate and test a pretrained detector.** Establish baseline precision/recall before fine-tuning.
4. **Fine-tune a small detector.** Use local examples and hard negatives.
5. **Implement calibration and ground-plane mapping.** Test against measured ground positions.
6. **Implement tracking and crossing geometry.** Add ball radius and temporal evidence.
7. **Add airborne trajectory fitting only where useful.** Validate against independently labelled clips.
8. **Build an evaluation suite.** Measure false GOAL rate, goal recall, uncertainty rate, and scenario-specific failures.
9. **Collect failure cases and retrain.** Keep the test set untouched.
10. **Only then choose deployment.** Compare on-device inference with server inference using actual target phones and connectivity constraints.

## Final recommendation

Start with **Python + OpenCV + Ultralytics YOLO + NumPy + SciPy**. Fine-tune a small football detector, then use tracking and calibrated geometry to make the GOAL / NO GOAL decision. Use physics fitting as a supporting method for airborne balls, not as a promise that a single 2D camera can always reconstruct exact 3D position.

The most important early deliverable is a labelled, diverse, independently verified local dataset and an evaluation pipeline—not a mobile app or a large model.
