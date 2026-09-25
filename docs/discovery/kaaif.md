# Virtual Goal Detection — Solution

The system receives a continuous **video frame stream** from a single camera positioned behind the goal.

Camera
↓
Video Frame Stream
  ↓
Football Detection & Tracking
  ↓
Football Location Estimation
  ↓
Virtual Goal-Plane Intersection
  ↓
GOAL / NO GOAL
  ↓
Record Short Replay


### 1. Football Detection & Tracking

* Detect the football currently involved in play.
* Use **AI/computer vision + temporal tracking** to identify the correct football.
* Use previous frames to estimate the ball's position when it is temporarily occluded or difficult to detect.
* Reject football-like objects that are not the ball.
* Ignore the ball when it is sufficiently far from the goal, where precise detection is unnecessary.
* Focus computational resources on the **goal-side region of interest**.

### 2. Football Location Estimation

Determine the football's **position relative to the virtual goal** from the camera's observations.

This is the primary research challenge and the exact method is discussed later.

### 3. Goal-Plane Crossing

Construct the virtual goal from the calibrated goal markers.

Determine whether the football's trajectory/position crosses the **2D/3D virtual goal boundary**, using geometric reasoning and, where useful, projectile/trajectory modelling.

The system must account for the ball's physical size rather than treating it as a single point.

### 4. Decision & Replay

If the ball satisfies the defined goal conditions the system gives **GOAL** with the confidence score, otherwise, **NO GOAL** also with confidence score.

For each detected goal (and close/uncertain events), automatically save a **short video clip** containing the relevant play for replay and verification.

## Football location estimation

...