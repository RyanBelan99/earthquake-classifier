# Earthquake vs. Explosion Classifier

Machine-learning classifiers — **built from scratch, no ML libraries** — that tell
**earthquakes apart from underground nuclear explosions** from two seismic
measurements (body-wave and surface-wave magnitude). Written for CSC 242
(Introduction to AI) at the University of Rochester.

This is the classic seismic-classification example from Russell & Norvig's *AIMA*.
Each data point is `body_wave, surface_wave, label`, where label `0` = earthquake
and `1` = explosion. There are a "clean" (linearly separable) set and a "noisy"
(overlapping) set.

The same problem is solved three ways, each implemented by hand:

| Approach | File | Verified accuracy (clean data) |
|---|---|---|
| **Perceptron** | `PerceptronClassifier.py` | ~97% |
| **Logistic regression** | `LogisticClassifier.py` | ~97% |
| **Neural network** (1 hidden layer, backprop) | `NeuralNetwork.py` | ~90–97% |

## Run it

```bash
cd src

# Perceptron + logistic regression (reproduces AIMA figures 18.16 / 18.18).
# Interactive menu, options 1-10; also runs on the 1984 house-votes dataset.
python3 main.py

# From-scratch neural network on the earthquake data:
python3 NeuralNetwork.py
# -> Accuracy before training: ~50%
#    Accuracy after training:  ~95%
```

Standard library only — no numpy, no scikit-learn, nothing. The point of the
assignment was to implement the learning algorithms themselves.

## What's implemented from scratch

- **Linear classifier core** (`LinearClassifier.py`): dot-product evaluation,
  the weight-update rule, a decaying learning-rate schedule, and accuracy /
  squared-error metrics. Perceptron and logistic regression subclass it and
  supply their own threshold + learning rule.
- **Neural network** (`NeuralNetwork.py`): hand-built `Neuron` and `Connection`
  objects, feed-forward evaluation, and **backpropagation** with a sigmoid
  (logistic) activation.

## Attribution

- Assignment from CSC 242 (Prof. George Ferguson, University of Rochester).
- The earthquake and house-votes datasets come from the course / the AIMA textbook.
- All classifier and neural-network code was written from scratch by me.

## Fixes since the original submission

The linear classifiers worked as submitted (debug prints have been removed). The
neural network needed real fixes to actually learn, which are good illustrations
of common from-scratch-ML pitfalls:

- **Gradient sign:** the output-layer error used `abs(target - output)`, which
  discards the sign so weights could only move one way. Now uses the signed error.
- **Weight initialization:** all weights started at `0`, making every neuron in a
  layer identical (symmetry) so nothing could differentiate. Now initialized to
  small random values.
- **Error accumulation:** hidden-layer errors were summed with `+=` but never
  reset between examples, so they compounded across the whole run. Now cleared
  each backprop pass.
- **Depth:** the demo built a 50-hidden-layer network, which can't train with
  plain sigmoid backprop (vanishing gradients). The demo now uses one hidden
  layer, and the network reads the bundled earthquake data instead of a data file
  that wasn't included.
