---
title: "Sim-to-Real Reinforcement Learning on the SO-101 Robot"
date: 2026-08-26T12:00:00
excerpt: "Simulation and real-robot rollouts of a vision-only pick-and-place policy driven solely by wrist-camera observations."
show_date: true
---

Some rollouts of the reinforcement learning policy trained by my group for the final project and competition in ETH Zürich’s Robot Learning course.

For the task, a cube of the target colour had to be picked up and placed into the bowl. At the start of each rollout, the positions of both the cube and the bowl were randomised. Only images from the wrist-mounted camera on the SO-101 robot arm were provided as policy observations. The policy was trained entirely in simulation using reinforcement learning.

We won first place in the task category.

## Rollouts in Simulation

<figure class="video-figure" markdown="0">
  <video autoplay controls loop muted playsinline preload="metadata">
    <source src="{{ '/assets/videos/sim_rollouts.mp4' | relative_url }}" type="video/mp4">
    Embedded MP4 video is not supported by this browser.
  </video>
  <figcaption>Parallel policy rollouts of the pick-and-place task in simulation. For each rollout, the view from the wrist-mounted camera is shown on the left, and an overview of the scene is shown on the right.</figcaption>
</figure>

## Rollout on the Real Robot

<figure class="video-figure" markdown="0">
  <video autoplay controls loop muted playsinline preload="metadata">
    <source src="{{ '/assets/videos/real_rollout.mp4' | relative_url }}" type="video/mp4">
    Embedded MP4 video is not supported by this browser.
  </video>
  <figcaption>A policy rollout on the physical SO-101 robot, shown in real time.</figcaption>
</figure>

## Rollout from Demo Day

<figure class="video-figure" markdown="0">
  <video autoplay controls loop muted playsinline preload="metadata">
    <source src="{{ '/assets/videos/demo_day_rollout.mp4' | relative_url }}" type="video/mp4">
    Embedded MP4 video is not supported by this browser.
  </video>
  <figcaption>A policy rollout recorded during demo day, shown in real time.</figcaption>
</figure>
