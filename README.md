# Risk-Annotated Scene Graphs for a Pick-and-Place Robot Arm

## What this project is

An industrial robot arm can run the same pick-and-place motion in two scenes that look almost identical, and only one of them is hazardous. A vial without its cork can spill during the transfer. A corrosive pictogram on its label decides how severe the spill would be. A knife offered blade-first to a hand is a different risk from the same knife offered handle-first. Generic object detection does not capture these small, local cues.

This thesis builds two things for a tabletop work cell:

1. **A scene graph pipeline** that builds and updates a graph of the work cell (objects, parts, states, labels, relations) from the robot's cameras and joint states, live or from recorded ROS bags.
2. **An agentic annotation framework** that inspects the graph and the images with tools (vision-language models, geometry, knowledge lookup) and writes risk-relevant annotations onto the graph, each with evidence and confidence.

The annotated graph is produced while the task runs. It does not control the robot.

## Example

A hazard is stored as a triple with its supporting information attached:

```yaml
(knife_03.blade, pointing_toward, human_hand_zone)
  hazard_type : mechanical_cut
  severity    : <level>
  evidence    : image region + graph edge + tool output
  confidence  : 0.0 to 1.0, or status "unknown"
```

Uncertain or contradictory observations are marked `unknown`. Annotations without valid evidence are rejected by schema validation.
