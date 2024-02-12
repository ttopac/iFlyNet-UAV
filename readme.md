# Estimating Flight State of Aerial Vehicle from Distributed Multimodal Data - Full-scale UAS

## Summary
Accompanying paper: T. Topac, C. Gray, and F.-K. Chang, “Fly-by-feel: Learning Aerodynamics from Multimodal Wing Mechanics,” in AIAA SciTech 2024 Forum, Jan. 2024. (https://arc.aiaa.org/doi/10.2514/6.2024-2403)

As an extension to [ttopac/iFlyNet-MW](https://github.com/ttopac/iFlyNet-MW), this project studies bio-inspired flight awareness of a full-scale UAS from wind tunnel tests. Real-time piloting variables and aerodynamic performance metrics of the aircraft are estimated through a model informed by static/dynamic stress state of one of the wings. The sensing data is collected by a distributed sensor network that seamlessly integrates onto the wing structure with near zero footprint, weight, and power consumption. 

## Code
This repository includes code for:
- (0_) Exploratory analysis and quality checks of the acquired data.
- (1_) Consolidation and synchronization of (i) sensor network SG/PZT, (ii) wind tunnel EDS, and (iii) Blackswift S0 UAS built-in instrumentation data for model training and dynamic evaluation.
- (2_) Training of the estimation model.
- (3_) Inference on the trained estimation model.
- (4_) Graphical twin interface for dynamic experiments.
- (5_) Reduced dataset training runs.
- (6_) Evaluation of the error metrics per flight condition.

## Data
Accompanying data will be made public at a later time.

## Contact
Feel free to reach out for comments, suggestions, and collaboration ideas.