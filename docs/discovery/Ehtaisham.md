# ⚽ Virtual Goal Detection

### Smart Goal Detection for Informal Football

## 📌 Overview

In informal football, two bricks are often used as goalposts instead of traditional goalposts and nets. This creates disputes when the ball crosses the goal area, especially during fast or airborne shots.

**Virtual Goal Detection** is a proposed system that uses **Computer Vision and Sensor Technology** to automatically detect and verify whether the football has crossed the goal area.

---

## 💡 Solution

The system explores two complementary approaches:

### 📷 Computer Vision

A single camera observes the goal area and uses AI/computer vision to:

- Detect the football.
- Track its movement across frames.
- Estimate its position and trajectory.
- Use the two physical markers to create a **Virtual Goal**.
- Analyse whether the football crosses the virtual goal plane.
- Classify the event as **GOAL, NO GOAL, or UNCERTAIN**.
- Save a short replay for verification.

### 🔌 Sensor-Based Detection

Sensors can be placed around the goal region to detect when a **spherical object** enters or crosses the defined goal area.

The sensor system can:

- Detect an object entering the goal region.
- Trigger a goal event.
- Maintain the goal count.
- Provide additional evidence to the vision system.

The sensor approach can work independently or be combined with computer vision for improved verification.

---

## 🔄 How It Works

```text
                    VIRTUAL GOAL SYSTEM
                           │
              ┌────────────┴────────────┐
              │                         │
          📷 CAMERA                 🔌 SENSORS
              │                         │
              ↓                         ↓
      Football Detection          Object Detection
              │                         │
              ↓                         ↓
       Football Tracking          Goal-Region Event
              │                         │
              ↓                         │
    Position & Trajectory              │
              │                         │
              ↓                         │
       Virtual Goal Analysis            │
              │                         │
              └────────────┬────────────┘
                           ↓
                    Evidence Analysis
                           ↓
              ┌────────────┼────────────┐
              ↓            ↓            ↓
            GOAL        NO GOAL     UNCERTAIN

Systam Architecture : 
┌─────────────────────┐
│      Camera         │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Football Detection  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Football Tracking   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Position Estimation │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Virtual Goal      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Crossing Analysis   │
└──────────┬──────────┘
           │
           ├───────────────┐
           ↓               ↓
      Sensor Data      Vision Data
           │               │
           └───────┬───────┘
                   ↓
          Decision Engine
                   ↓
       GOAL / NO GOAL / UNCERTAIN
                   ↓
                Replay
            
                           │
                           ↓
                     Short Replay
