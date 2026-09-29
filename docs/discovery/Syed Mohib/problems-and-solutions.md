# Goal Detection — Problems & Solutions

## 1. Lighting conditions affect ball detection

**Problem:**
A ball detector trained on bright daylight footage may fail at night or in shadows.

**Solutions:**
- Train/test the ball detector on a variety of lighting conditions from the start, not just one type of footage.
- Preprocess images (auto-adjust brightness/contrast) so night/shadow frames look more like daylight frames before detection.
- Use a camera with better low-light sensors (infrared or good low-light phone cameras).
- Use synthetic data augmentation — artificially darken/brighten existing daylight footage to simulate different lighting cheaply.
- Add a fixed lighting setup (e.g., a cheap LED floodlight) behind the goal to keep lighting consistent.
- Add a confidence-based fallback — if detection confidence drops too low, return "UNCERTAIN" instead of guessing.

---

## 2. Camera not centered behind the goal

**Problem:**
If the camera is placed slightly to one side rather than dead-center behind the goal, the whole geometry skews, and left/right accuracy suffers.

**Solutions:**
- Enforce a fixed setup position (mark it physically) or calculate the camera's off-center angle during calibration and correct for it mathematically.
- Detect the ground line between the two markers (not just the two points) to get more information for auto-correcting skew.
- Use a printed calibration mat or marker (like a checkerboard or QR sticker) on the ground so the software can instantly compute the camera's exact angle and position.
- Use the phone's built-in gyroscope/accelerometer to detect tilt, separating "camera tilt" from "camera off-center."
- Show a live on-screen alignment guide during setup (like a level tool) so the user visually centers the camera correctly before recording.
- Use multiple-angle averaging with assumed symmetric error correction based on known goal width, if slight miscentering is unavoidable.

---

## 3. Different balls have different sizes

**Problem:**
A futsal ball, a size-4 ball, and a size-5 ball are all different real diameters — if the system assumes one fixed ball size for depth calculations, using the wrong ball breaks the math.

**Solutions:**
- Let the user select/enter the ball type/size before starting, or auto-calibrate using a known reference frame at the start.
- Ask the user to place any object of known size (a shoe, a bottle) at the goal line briefly before play starts, and use that for scale calibration instead of relying on ball type.
- Use the first clear detection of the stationary ball (e.g., before a kick) to measure its pixel size and set that as the calibration reference for the rest of the session.
- Train the detector to automatically classify ball type (futsal vs size-4 vs size-5) from its appearance, removing the need for manual input.
- Cross-check ball-based depth estimates against the known goal-marker distance as a sanity check.
- Fall back to marker-based geometry (instead of ball-size-based depth) for frames where the ball is only partially visible and size calibration seems unreliable.
