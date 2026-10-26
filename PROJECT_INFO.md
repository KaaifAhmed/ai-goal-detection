# Project Research Starting Point
## Portable Virtual Goal Detection for Informal Football

### Core idea

Informal football games often use two bricks, stones, shoes, or other objects to mark the two sides of a goal rather than actual goalposts and a crossbar.

This creates ambiguity when the ball travels through the goal area **above the ground**. Players may disagree about whether the ball passed through the intended goal or over/around it.

The proposed system is a **single-camera, goal-side computer-vision system**.

A camera positioned behind the goal observes the play. The system should:

1. Identify the two physical goal markers.
2. Establish the corresponding virtual goal geometry.
3. Detect and track the football.
4. Estimate the ball's movement through the scene.
5. Determine whether the ball crossed the defined goal region.
6. Produce a **GOAL / NO GOAL / UNCERTAIN** decision.
7. Ideally retain the relevant video for replay.

### Important research principle

We should **not decide beforehand** that:

- YOLO will detect the ball,
- homography will solve the geometry,
- projectile equations will reconstruct the trajectory,
- monocular depth will solve the 3D problem,
- or some particular neural network will be used.

Those are **candidate techniques to investigate**.

The research question is:

> **What information about a football's 3D position and trajectory can reliably be recovered from a single ordinary camera looking toward an informal goal, and what combination of computer vision, geometry, physics, and machine learning can turn that information into a reliable goal decision?**

---

# Phase 1 — Define the Problem Precisely

### Objective

Turn the intuitive idea of "did the ball go into the goal?" into a mathematically and computationally precise problem.

### Research questions

**1. What exactly constitutes a goal in our system?**

We need to define:

- goal width
- goal height
- goal plane
- ball radius
- whether the entire ball must cross the plane
- what happens if the ball hits a marker
- what happens if it passes above the goal
- what happens if it passes outside one side
- what happens if it crosses the plane and comes back

We should initially define our own **virtual-goal rules**, rather than assuming FIFA rules automatically apply to informal football.

**2. What does the camera actually observe?**

Separate:

### Directly observable

- 2D pixel location
- apparent ball size
- apparent ball shape
- motion between frames
- goal-marker locations
- players/objects
- shadows, lighting, etc.

### Potentially inferable

- distance/depth
- real-world ball position
- velocity
- trajectory
- goal-plane crossing

This distinction is fundamental.

**3. What level of accuracy is actually required?**

For example:

> If the ball is clearly 50 cm above the goal, accuracy doesn't matter much.

But:

> If the ball is 2 cm above the imaginary crossbar, accuracy matters enormously.

We should therefore investigate **error tolerance around the boundaries**, rather than simply targeting generic "90% accuracy."

---

# Phase 2 — Investigate the Single-Camera Problem

This is probably the most important research phase.

### Objective

Determine what can and cannot be reconstructed from **one normal camera**.

Research:

### A. Monocular vision

Investigate:

- monocular depth estimation
- camera calibration
- perspective
- vanishing points
- camera pose estimation
- world coordinates vs image coordinates
- scale ambiguity
- 2D → 3D reconstruction

The key question:

> **Can we obtain sufficiently accurate 3D information from one camera when we know something about the physical environment?**

That last part is important.

We don't have an arbitrary scene.

We have:

> ground + two goal markers + a known/estimated goal height + a relatively fixed camera.

That may provide useful constraints.

---

### B. Camera calibration

Research:

- intrinsic parameters
- focal length
- principal point
- lens distortion
- extrinsic parameters
- camera position/orientation

We should determine:

> How much calibration is required for our system to work?

And:

> Can calibration be performed automatically using the goal markers?

---

### C. Perspective geometry

Research:

- perspective projection
- projective geometry
- homography
- planar transformations
- vanishing points
- line detection

Particularly:

> Can the ground/goal geometry be recovered from the image using known points or lines?

Do **not** decide yet that homography is the answer. Investigate what it can and cannot provide.

---

# Phase 3 — Research Football Detection & Tracking

### Objective

Determine how reliably we can identify the football in real match footage.

Research:

### Ball detection

Compare approaches such as:

- traditional computer vision
- color/shape-based detection
- optical flow
- object detection
- YOLO-family detectors
- segmentation
- custom-trained football detector

Questions:

> How small can the football become in the image before detection fails?

> How does motion blur affect detection?

> What happens when a player partially blocks it?

> What happens when the ball is directly behind a player?

> Does a generic pretrained detector work, or do we need football-specific training?

---

### Ball tracking

Research:

- Kalman filtering
- optical flow
- object tracking
- trajectory smoothing
- multi-frame tracking
- prediction through temporary occlusion

The interesting question is:

> If AI misses the ball for 2–5 frames, can mathematics/tracking recover its likely trajectory?

This fits perfectly with our principle of **using AI for perception rather than making AI responsible for the final decision**.

---

# Phase 4 — Investigate 3D Position From One Camera

This should be treated as its own research problem.

### Objective

Find possible methods for estimating:

> **Where is the ball in real-world 3D space?**

Research several competing approaches.

### Approach A — Known scene geometry

Can we derive depth from:

- ground plane
- goal markers
- camera calibration
- known goal dimensions
- known camera height
- perspective?

### Approach B — Ball-size/depth estimation

The physical football has approximately known dimensions.

Could apparent ball size provide a depth estimate?

Investigate:

> How sensitive is this to image resolution, detection error and perspective?

### Approach C — Monocular depth models

Investigate modern monocular depth estimation.

Question:

> Can an off-the-shelf depth model estimate football depth accurately enough for this particular task?

Don't assume yes.

### Approach D — Motion-based reconstruction

Research:

- optical flow
- structure from motion
- temporal geometry

Question:

> Can the ball's movement itself provide useful depth information?

### Approach E — Physics constraints

Investigate whether known/estimated:

- ball radius
- gravity
- frame timestamps
- trajectory
- bounce
- launch direction

can constrain the possible 3D trajectory.

Again, **don't assume projectile motion will solve it**. A kicked football is affected by:

- spin
- air resistance
- Magnus effect
- bounce
- collisions
- deflections

But simplified physics may still be useful as a constraint.

10203040506070510152025Horizontal distance (m)Height (m)

Rises 6.8 m · travels 32.6 m · airborne 2.4 s

Initial speed

m/s

Launch angle

°

Give feedback

The research question is therefore:

> **How much can physics improve a noisy computer-vision estimate?**

---

# Phase 5 — Research Goal Geometry

### Objective

Determine how the system should mathematically represent the "invisible goal."

Possible representations to investigate:

### Model 1 — 2D image rectangle

Simplest.

```text
      ┌────────────┐
      │            │
      │    GOAL    │
      │            │
      └────────────┘
```

Question:

> Is a 2D image-space solution actually sufficient under some assumptions?

---

### Model 2 — 3D goal plane

```text
       │
       │
       │
       │ GOAL
       │ PLANE
       │
       │
───────┼──────── ground
```

Question:

> Can we establish a real-world coordinate system and calculate whether the ball crosses this plane?

---

### Model 3 — 3D goal volume

Instead of treating the goal as merely a plane, model:

> width × height × ball diameter

Then investigate the correct mathematical condition for the **whole ball** to cross.

---

### Model 4 — probabilistic goal region

Because our measurements will contain errors, perhaps the system shouldn't say:

> "The ball is exactly here."

Instead:

> "Given our uncertainty, there is a 97% probability that the ball crossed the goal boundary."

This could eventually lead to:

**GOAL**

**NO GOAL**

**UNCERTAIN**

rather than forcing a false certainty.

---

# Phase 6 — Build Software Using Real Football Video

Only now do we start implementing.

## Stage 6A — Controlled goal-view dataset

Before Karachi footage, create a dataset specifically for this problem.

Find/create goal-view videos containing:

- ground shots
- airborne shots
- shots directly toward camera
- shots at different angles
- balls passing over the goal
- balls passing beside the goal
- balls crossing the goal
- near misses
- balls hitting players
- balls hitting goal markers
- deflections
- multiple players
- different lighting

Each event should have a human-labelled ground truth:

```text
GOAL
NO GOAL
UNCERTAIN
```

And ideally additional labels:

```text
ball trajectory
goal boundaries
goal crossing frame
```

---

## Stage 6B — Ball detection only

Forget goal decisions.

Build:

> **Video → football location**

Measure:

- detection accuracy
- missed detections
- false detections
- performance under occlusion
- performance under motion blur
- FPS
- latency

---

## Stage 6C — Ball tracking

Then:

> **Video → continuous football trajectory**

Test whether tracking can maintain the ball when detection temporarily fails.

---

## Stage 6D — Goal geometry

Now implement candidate methods for establishing the virtual goal.

Test them independently.

Don't combine everything yet.

---

## Stage 6E — Goal decision

Finally:

```text
Video
 ↓
Ball detection
 ↓
Ball tracking
 ↓
Position/trajectory estimation
 ↓
Virtual goal geometry
 ↓
Decision algorithm
 ↓
GOAL / NO GOAL / UNCERTAIN
```

The important thing is that we should be able to inspect every intermediate stage.

---

# Phase 7 — Test on Karachi Football Videos

Only after the basic software works on controlled/general footage.

### Objective

Find out what happens in the actual environment we're targeting.

Collect footage from **real Karachi football games**.

We need diversity:

- different grounds
- concrete/turf/grass
- day
- evening
- floodlights
- different cameras/phones
- different goal-marker objects
- different goal sizes
- different camera positions
- different playing styles

Then compare:

> **Controlled dataset performance**

against

> **Karachi real-world performance**

This is where we will probably discover problems that aren't obvious in laboratory testing.

---

# Phase 8 — Failure Analysis

This deserves an entire phase rather than being treated as an afterthought.

For every incorrect decision, classify the failure.

For example:

| Failure | Possible cause |
|---|---|
| Ball not detected | Motion blur |
| Ball lost | Player occlusion |
| Wrong depth | Monocular ambiguity |
| Wrong goal geometry | Marker detection |
| False goal | Trajectory error |
| Missed goal | Ball outside camera view |
| Wrong decision | Camera perspective |
| Wrong decision | Lighting |
| Wrong decision | Camera too far/close |
| Wrong decision | Goal marker moved |

Then ask:

> **Which problems can software solve?**

> **Which require better calibration?**

> **Which require different camera placement?**

> **Which are fundamentally impossible with one camera?**

That final question is crucial.

If we discover that some cases are fundamentally ambiguous from one viewpoint, the product shouldn't pretend otherwise.

That is where the **UNCERTAIN** state becomes valuable.

---

# Phase 9 — Determine the Minimum Hardware Requirements

Only after the software tells us what it needs.

Instead of saying:

> "Let's buy a Jetson."

we ask:

### Camera requirements

Research:

- resolution
- FPS
- shutter speed
- global vs rolling shutter
- low-light performance
- field of view
- focal length
- autofocus vs fixed focus
- distortion

### Computing requirements

Measure the software's actual:

- CPU usage
- GPU usage
- RAM
- inference time
- power consumption

Then determine the cheapest hardware capable of running it.

---

# Phase 10 — Hardware Prototype

Now build the first physical system.

Something like:

```text
             🧱             🧱
              \             /
               \           /
                VIRTUAL
                 GOAL
                   │
                   │
                   📷
                   │
              COMPUTE UNIT
                   │
                Wi-Fi
                   │
                 PHONE
```

The hardware prototype should initially prioritize:

**function > appearance > miniaturization**

It should answer:

> Can a real person place this behind a goal, calibrate it, and get reliable decisions during an actual game?

---

# Phase 11 — Real-World Field Testing

Now test the complete system.

Measure:

### Detection

- ball detection rate

### Decision

- goal accuracy
- false goals
- missed goals
- uncertain decisions

### Performance

- processing latency
- FPS
- battery life

### Usability

- setup time
- calibration time
- camera placement tolerance
- frequency of recalibration

### Robustness

- day/night
- weather
- different grounds
- different marker objects
- different goal sizes
- different phones/cameras

---

# Phase 12 — Product Research

Only once the technical feasibility is established.

Then research:

- typical football grounds in Karachi
- how people currently mark goals
- typical goal dimensions
- where players put cameras/phones
- acceptable device size
- expected price
- whether players would buy vs rent
- football academies
- amateur tournaments
- university football
- corporate football
- futsal
- local sports facilities

This determines whether we're building:

> **a personal consumer device**

or perhaps:

> **a service for football grounds/tournaments.**

---

# The Research Questions We Should Keep Coming Back To

I would put these at the top of the project's research document.

### **RQ1 — Perception**
> How reliably can a single ordinary camera detect and track a football during real football play from behind the goal?

### **RQ2 — Geometry**
> What information about the physical goal and surrounding environment can be recovered from one camera?

### **RQ3 — Depth**
> Can the ball's 3D position relative to the goal plane be estimated accurately enough using a single camera and environmental constraints?

### **RQ4 — Physics**
> Can temporal information and physical models of football motion improve the estimate of the ball's position/trajectory?

### **RQ5 — Decision**
> Given uncertain measurements, what mathematical model can determine whether the ball crossed the virtual goal region?

### **RQ6 — Uncertainty**
> When is a goal decision fundamentally ambiguous, and how can the system recognize and communicate that uncertainty?

### **RQ7 — Robustness**
> How does the system behave under real Karachi conditions: night lighting, occlusion, motion blur, different surfaces, different goal-marker positions, and different camera placements?

### **RQ8 — Hardware**
> What is the minimum camera and computing hardware required to run the validated software in real time?

---

# Our Architecture Should Emerge From the Research

I think this should be one of the project's rules:

> **We don't choose the architecture first. We discover it through experiments.**

But conceptually, we're aiming toward something like:

```text
                    VIDEO
                      │
                      ▼
              ┌───────────────┐
              │   PERCEPTION  │
              │               │
              │ AI / CV       │
              │               │
              │ • Ball        │
              │ • Markers     │
              │ • Players     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   TRACKING    │
              │               │
              │ Temporal data │
              │ Motion        │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   ESTIMATION  │
              │               │
              │ Geometry      │
              │ Camera model  │
              │ Depth         │
              │ Physics?      │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │    GOAL       │
              │    MODEL      │
              │               │
              │ Virtual 3D    │
              │ goal region   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │    DECISION   │
              │               │
              │ GOAL          │
              │ NO GOAL       │
              │ UNCERTAIN     │
              └───────────────┘
```

**AI is primarily responsible for perception.**

**Mathematics/geometry is responsible for spatial reasoning.**

**Physics is a candidate source of constraints.**

**A deterministic decision layer makes the final call.**

That's a very good engineering philosophy for this project—but whether each component belongs there should be established experimentally.

---
