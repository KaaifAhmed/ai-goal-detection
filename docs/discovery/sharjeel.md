# Virtual Goal Detection: Technical Flaws and Solutions

Project: single-camera system that decides GOAL / NO GOAL / UNCERTAIN for informal football using two physical markers as goalposts.

---

# 1. The ball is not always seen (it is small, fast, blurry, hidden by a player or cut off at the edge of the picture)

### Solution
- Record at 60 fps if the phone allows it (30 fps as a fallback), with a fast shutter.
- Lock focus and exposure, and turn off stabilisation.
- Use a tracker (Kalman filter) to fill gaps of a few frames.
- Estimate the crossing point between two frames from the ball's path when the shot is very fast.
- If the ball is hidden or cut off exactly at the crossing moment, output UNCERTAIN instead of guessing.

---

# 2. The ball's size can mislead (a ball close to the camera looks bigger, and a ball in front of the goal can look like it is inside it)

### Solution
- Don't decide from the 2D picture alone.
- During setup, place a real ball on the goal line and record how big it looks.
- Count a goal only if the ball is inside the goal points and its size is close to that reference size.
- Average the size over several frames around the crossing, and use a generous tolerance.
- If it is inside the points but clearly too big or too small, output NO GOAL, or UNCERTAIN if it is close.

---

# 3. Camera distance and angle change the accuracy (too far and the ball is tiny; too close and it is blurry, cut off or hit by shots; a sideways angle distorts the view)

### Solution
- Use these starting values, then test them (they are estimates):

| Setting | Start with |
|---|---|
| Distance behind the goal line | 3 to 5 m (avoid more than about 6 m) |
| Height | 1.5 to 2 m, tilted slightly down |
| Angle | Centred, or at most about 15 to 20° off-centre |
| Framing | The goal fills about 40 to 60% of the frame width |
| Frame rate | 60 fps if possible (30 fps as a fallback) |
| Camera settings | Focus and exposure locked, stabilisation off |

- Rough size guide (estimate for 1080p): the ball is about 70 px wide at 4 m and about 40 px wide at 8 m.
- Record clips at two or three distances early and choose the one where the ball is detected best.
- Keep the angle small; a large angle makes the goal points and the size check less reliable.
- Mount the camera high (on a fence, wall or pole), offset slightly to one side, or behind a net so shots are less likely to hit it.

---

# 4. The camera or a marker gets moved (everything is measured from the setup, so a shifted view breaks the results)

### Solution
- Keep the camera fixed on a tripod, wall or pole.
- Re-tap the markers whenever the setup changes.
- Detect a shifted view (for example, fixed background points that have moved) and ask for recalibration.

---

# 5. Close calls are the hardest (the system is least accurate when the ball is right at the edge)

### Solution
- Output UNCERTAIN near the boundaries.
- Show the clip with the goal points, ball path and crossing frame drawn on it, so humans can judge.
- Log every disagreement as a labelled case for testing and retraining.
- Report accuracy separately for clear cases and close cases.

---

# 6. Long recordings are too big to analyse, and stopping to send a clip can miss the next play

### Solution
- Never stop recording; keep a rolling buffer of the last ~15 seconds.
- Use a button to save and send only that clip when a goal is disputed.
- Process the clip in the background while recording continues.
- This also covers the case where the system never triggered on its own: the last seconds are always in the buffer.
- Consider an automatic trigger later (a light check for the ball near the goal area).

---

# 7. There is no ready-made model for this (no model gives goal decisions out of the box)

### Solution
- Use a pre-trained YOLO model (nano or small) only to find the ball.
- Write the decision logic yourself as simple geometry code.
- Test the pre-trained model first; fine-tune it only if the detection rate is low.
- Fine-tune with Ultralytics on free Google Colab or Kaggle GPUs (about an hour).

---

# 8. Not enough example videos (real goal-view footage is rare, and labelling takes time)

### Solution
- Record your own clips early, including goals, near misses, over-the-bar shots and blocked shots.
- Label roughly 500 to 1,500 frames with Roboflow or CVAT, and add a public football dataset from Roboflow Universe if needed.
- Split train and test by clip, not by frame, and keep about 20% of clips aside as a test set.

---

# 9. Slow computers (the team's laptops may not have a strong GPU)

### Solution
- Process clips offline, not live.
- Train once on free cloud GPUs, then run on a normal laptop CPU.
- Use a small model and process only short clips.

---

# 10. Difficult lighting (evening, floodlights and shadows make the ball hard to see)

### Solution
- Test in daylight first.
- Add evening and floodlit clips later and report the results separately.
- State clearly which conditions are supported.
