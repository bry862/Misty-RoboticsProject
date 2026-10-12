# Misty Companion Pet
A robotic companion pet built on the Misty platform. Misty notices people nearby, tracks their face, and responds to simple hand gestures with pet-inspired reactions.

## Overview
Misty uses its camera and built-in sensors to perceive its surroundings: whether a person is present, how close they are, whether other people are around, and whether there are obstacles between Misty and the person. When Misty detects a person, it turns to look at them and reacts with expressive behaviors such as changing its eyes, moving its head and arms, and playing sounds. People can also direct Misty with hand gestures. For example, a wave triggers a greeting and a raised open palm tells Misty to stop.

## Team Members
| Name               | GitHub Handle | Contribution              | 
|--------------------|---------------|---------------------------|
| Angelina Kovalchuk | @Angelinakoval | |
| Anna Muravyeva     | @AnnaMuravyeva | |
| Brayhan Morrovel   | @bry862| |
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


## System Architecture

**Behavior state machine**

```
IDLE → NOTICED (face seen) → ENGAGED (tracking the person) → REACT (gesture or greeting) → COOLDOWN → IDLE
```

Any state can move to `STOPPED` when the raised-palm gesture or a safety sensor is triggered. The `COOLDOWN` state keeps Misty from pestering the same person.


## Planned Capabilities by Checkpoint
| Checkpoint | Focus |
|------------|-------|
| 1 (Oct 15) | Face detection and head tracking, pet reactions, wave and stop gestures, safety stop |
| 2 (Nov 5) | Following a person at a safe distance, "point to go" gesture, mapping and navigation |
| 3 (Nov 24) | User testing with Hunter students, refined reactions, responsible-design review |
| Final | Integrated companion behavior and full evaluation |


## Setup and Running
**Requirements**
- Misty II on the same Wi-Fi network as the laptop
- Python 3.10 or newer

