# AI-Project-G17

## Task 1: group members

- Edouard de Noblet, Mechanical Engineering, edenoblet@gmail.com
- Mohamed Ali Kabiri, Computer Science, mohamedali.kabiri@gmail.com
- Lorenzo Soubiran Frixione, Computer Science, lorenzo.soubiranfrixione@edu.ece.fr
- Zineb El Koussaimi, Computer science, koussaimizineb@gmail.com

## Research idea

**Title:** Multi-horizon Forecasting of Switch Queue Occupancy from High-Frequency Telemetry, with Prediction Intervals

**In simple words:** we predict how full the queue of a network switch will be a few milliseconds from now, and how sure we are about it.

### Proposal (Option A)

**The problem.** Packets wait in a switch like people in a checkout line. When competing traffic turns on, this queue can fill within a few milliseconds. Packets are then delayed or lost. Reacting only once the queue is full is too late, so we want to predict it before it happens.

**The data.** We use recordings that were already collected on an emulated network testbed (BMv2 software switches in Mininet). Every 2 ms (500 times per second), a small measurement packet reports the depth of the queue it leaves and how long it waited there. It does this at two switches: the bottleneck switch, and a second switch after it (the downstream switch).

**The task.** Using the measurements up to now, we predict the queue depth at the bottleneck switch at four moments in the future (the horizons):

| samples ahead | time ahead |
|---:|---:|
| 1 | 2 ms |
| 5 | 10 ms |
| 10 | 20 ms |
| 25 | 50 ms |

For each horizon we give two things:

- **one number:** our best guess (the point forecast);
- **a range:** the 0.1, 0.5 and 0.9 quantiles. The true value should be between the 0.1 and the 0.9 quantile about 80% of the time.

### What we will do

We follow a Titanic-style walkthrough:

1. **Explore the data.** Look at the distributions, the autocorrelation (how similar the queue is to its own recent past), how predictions get harder further ahead, and the lost samples.
2. **Prepare the features.** Turn the recordings into examples of "past measurements, future queue" (supervised windows). The features are lags, moving averages, slopes, rolling variance and the readings of the downstream switch.
3. **Train the models, from simple to complex.** A statistical model, gradient-boosted trees, a recurrent network, and an attention or TCN model. We compare them with two baselines: persistence (the queue stays as it is now) and EWMA (a weighted mean where recent samples count more).
4. **Evaluate the models.** For each horizon we measure:
   - MAE and RMSE: how far the point forecast is from the truth;
   - pinball loss: how good the 0.1, 0.5 and 0.9 quantiles are;
   - coverage: how often the truth falls inside the 80% interval.

**Keeping it fair.** We split train, validation and test by recording, so that no recording is in two groups (this avoids leakage). The test recordings are used only once, at the end.

### Why it matters

Knowing how full a queue will be, and how sure we are about it, is a building block for networks that manage themselves.
