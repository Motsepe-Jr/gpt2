# Evaluating Learning Rate Scheduling Techniques based on Andrej Karpathy's GPT-2

![Learning Rate](/assets/schedules.png)

## Overview
This repository contains code to evaluate various learning rate scheduling techniques based on AK's GPT-2. Inspired by Karpathy's use of the Cosine Schedule, I attempt to test different schedulers  to assess their impact on performance, convergence speed, and training stability.

## Learning Rate Schedulers Implemented
- **ConstantLR**: Keeps learning rate constant.
- **StepLR**: Reduces learning rate at regular intervals.
- **MultiStepLR**: Reduces learning rate at specified milestones.
- **LinearLR**: Decays learning rate linearly over training.
- **ExponentialScheduler**: Applies exponential decay to the learning rate.
- **CosineScheduler**: Uses cosine annealing with a warmup phase.
- **PolynomialLR**: Decays learning rate according to a polynomial function.
- **OneCycleLR**: Cycles learning rate up and down with warmup and cooldown.
- **CosineAnnealingWarmRestarts**: Applies cosine annealing with periodic restarts.
- **CyclicLR**: Oscillates learning rate between minimum and maximum values.

For detailed results and visualizations, check out the accompanying blog post [here](https://github.com/Motsepe-Jr/gpt2).

![Learning Rate](/assets/loss.png)

## Results
- **Best Training Stability**: OneCycleLR (0.9388)
- **Lowest Final Loss**: OneCycleLR (3.355333)
- **Fastest Scheduler**: CosineAnnealingWarmRestarts

## References
- Murphy, Kevin P. (2012). *Machine Learning: A Probabilistic Perspective*. Cambridge: MIT Press.
- Andrej Karpathy GPT-2: https://github.com/karpathy/build-nanogpt

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

