# Misty Companion Pet
A robotic companion pet built on the Misty platform. Misty notices people nearby, tracks their face, and responds to simple hand gestures with pet-inspired reactions.

## Overview
Misty uses its camera and built-in sensors to perceive its surroundings: whether a person is present, how close they are, whether other people are around, and whether there are obstacles between Misty and the person. When Misty detects a person, it turns to look at them and reacts with expressive behaviors such as changing its eyes, moving its head and arms, and playing sounds. People can also direct Misty with hand gestures. For example, a wave triggers a greeting and a raised open palm tells Misty to stop.

## Team Members
| Name               | GitHub Handle | Contribution              | 
|--------------------|---------------|---------------------------|
| Angelina Kovalchuk | @Angelinakoval | Gesture perception (STOP, WAVE, COME, GO_AWAY), combined demo |
| Anna Muravyeva     | @AnnaMuravyeva | Pet personality and touch interactions |
| Brayhan Morrovel   | @bry862| Face recognition and head tracking |
| Calvin Lin         | @lincalvin76 | |
| Elion Prelvukaj    | @Prelvukajelion | |


## The Problem and Opportunity
Owning a pet can provide companionship. But owning a pet comes with responsibilities not many people can take on or would like to commit to. Students and staff on a busy campus often have no way to enjoy the brief, low-pressure moments of connection that a pet provides.

We are not trying to build a robot that replaces a real pet. Instead, we are exploring how a robot can perceive its environment and the people in it, including their body language, and respond in a way that feels engaging and companion-like. Our central question is how a robot can read human presence and gestures and react in ways that feel friendly while respecting people's space and attention. 


## Stakeholders and Context of Use
**Primary stakeholders.** Hunter students and staff who choose to interact with Misty.

**Secondary stakeholders.** People in the same space who are not interacting with Misty but may still be affected by its movement or sounds, such as students studying nearby or people walking past.

**Context.** Misty is intended for shared campus spaces like a lab, lounge, or hallway. Interactions are brief: a person approaches, Misty notices them, and it responds based on current conditions. Because the setting is shared and public, Misty's behavior must be non-disruptive, non-intrusive, and safe, and it must respect personal space.

## Requirements
**Functional requirements**

| # | Misty must... |
|---|---------------|
| F1 | Detect a person's face and turn her head to look at the closest person within about 1 m |
| F2 | Remember people she has seen before and recognize them again |
| F3 | Recognize four hand gestures: STOP, WAVE, COME and GO_AWAY |
| F4 | React to each gesture: stop (STOP), greet (WAVE), drive toward the person while keeping eye contact (COME), back up, say goodbye and turn around (GO_AWAY) |
| F5 | React to head and chin pats with pet-like behaviors: speech, lights, and movement |

**Non-functional requirements**

| # | Requirement |
|---|-------------|
| N1 | **Safety.** STOP halts driving and head movement right away. The one exception is GO_AWAY, a short back-up and turn that always finishes. Misty never keeps driving after the program closes |
| N2 | **Responsiveness.** Face recognition runs in the background, so it never delays reacting to a gesture |
| N3 | **Not pestering.** The same gesture can't trigger again within 4 seconds, and a wave can't interrupt a movement that is still happening |
| N4 | **Respect for space.** Misty only follows faces within about 1 m, unless the person calls her over with COME |
| N5 | **Privacy.** Saved face data stays on the laptop and is never uploaded to the repository |
| N6 | **Robustness.** If face recognition can't load, Misty still responds to gestures and pats |


## System Architecture

**Behavior state machine**

```
IDLE → NOTICED (face seen) → ENGAGED (tracking the person) → REACT (gesture or greeting) → COOLDOWN → IDLE
```

Any state can move to `STOPPED` when the raised-palm gesture or a safety sensor is triggered. The `COOLDOWN` state keeps Misty from pestering the same person.

**Components**

| Folder | What it does |
|--------|--------------|
| `MistySkills/GesturePreception` | Finds the hand in each camera frame (MediaPipe) and recognizes STOP, WAVE, COME and GO_AWAY |
| `MistySkills/Greeting` | Recognizes faces (InsightFace), remembers new people and turns Misty's head toward the closest face |
| `MistySkills/PetInteractions` | Picks a pet-like reaction when Misty's head or chin is touched |
| `MistySkills/gesture_demo.py` | Runs all three together on one camera stream and sends Misty her reactions |
| `tests` | Automated tests for gesture recognition |


## Planned Capabilities by Checkpoint
| Checkpoint | Focus |
|------------|-------|
| 1 (Oct 15) | Face detection and head tracking, pet reactions, wave and stop gestures, safety stop |
| 2 (Nov 5) | COME and GO_AWAY gestures, eye contact while Misty drives toward a person, goodbye and turn-around on GO_AWAY, face tracking and gestures on one camera stream |
| 3 (Nov 24) | Stopping at a safe distance from the person, user testing with Hunter students, refined reactions, responsible-design review |
| Final | Mapping and navigation, integrated companion behavior and full evaluation |


## Setup and Running
**Requirements**
- Misty II on the same Wi-Fi network as the laptop
- Python 3.10 or newer

**Install** (once, from the project folder)
```
pip install -r requirements.txt
```

**Run the demo** (from the `MistySkills` folder)
```
python gesture_demo.py --ip <misty-ip>                # Misty's camera, all features
python gesture_demo.py --ip <misty-ip> --no-faces     # without face tracking
python gesture_demo.py --ip <misty-ip> --webcam       # laptop webcam, Misty still reacts
python gesture_demo.py --webcam --no-robot            # no robot: just prints the gestures it sees
```
In the video window, press `h` for a head pat, `c` for a chin pat, or `q` to quit.

**Run the tests** (from `MistySkills/GesturePreception`)
```
python -m unittest discover -s ../../tests -t ../..
```

**Privacy:** Misty saves the faces she sees to `MistySkills/Greeting/known_faces.pkl` so she can recognize people again. This file stays on the laptop and is never uploaded to GitHub. Delete it to make her forget everyone.

