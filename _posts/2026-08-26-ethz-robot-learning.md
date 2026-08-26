---
title: "Vision-Only Reinforcement Learning on the SO-101"
date: 2026-08-26T12:00:00
excerpt: "A vision-only policy for coloured-cube pick-and-place, with rollouts shown in simulation and on a real SO-101 robot."
show_date: true
---

Some rollouts of the reinforcement learning policy that my group trained for the final project and competition in the Robot Learning course at ETHZ. The task consisted of picking and placing a target colour cube in the bowl. The location of the cube and bowl were not fixed. The policy is vision-only: only images from the wrist camera attached to the S0-101 robot arm were provided as policy observations. Training was performed fully in simulation using reinforcement learning.

We won first place in the competition in our task category.

## Rollouts in Simulation

<figure class="video-figure">
  <video controls loop muted playsinline preload="metadata">
    <source src="{{ '/assets/videos/sim_rollouts.mp4' | relative_url }}" type="video/mp4">
    Embedded MP4 video is not supported by this browser.
  </video>
  <figcaption>Parallel policy rollouts of the pick-and-place task in simulation.</figcaption>
</figure>

## Rollout on the Real Robot

<figure class="video-figure">
  <video controls loop muted playsinline preload="metadata">
    <source src="{{ '/assets/videos/real_rollout_1.mp4' | relative_url }}" type="video/mp4">
    Embedded MP4 video is not supported by this browser.
  </video>
  <figcaption>A vision-only policy rollout on the physical SO-101 robot.</figcaption>
</figure>
