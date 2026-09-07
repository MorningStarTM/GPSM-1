"""3D SMPL-X simulator for GPSM — live/offscreen rendering of StateMachineGPT rollouts.

Sits outside src/gpsm on purpose: it is a visualization/testing tool built on
top of the trained model, not part of the model or training pipeline itself.
See simulator/live_viewer.py (interactive) and simulator/offscreen_test.py
(headless self-test).
"""
