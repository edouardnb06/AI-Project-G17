# AI-Project-G17

Task 1: group members
- Edouard de Noblet, Mechanical Engineering, edenoblet@gmail.com
- Mohamed Ali Kabiri, Computer Science, mohamedali.kabiri@gmail.com
- Lorenzo Soubiran Frixione, Computer Science, lorenzo.soubiranfrixione@edu.ece.fr
- Zineb El Koussaimi, Computer science, koussaimizineb@gmail.com

Task 3: Research idea
Title: Forecasting Network Congestion Before It Happens: Early-Warning Classification on Queue Telemetry

Proposal (Option A):
Network switches sometimes fill their buffers in short bursts, causing delay spikes and packet loss. Detecting this after it starts is too late to react, so we will forecast it. The dataset is a set of already-collected, high-frequency (1 kHz) queue-depth traces from an emulated network testbed. The task: given a sliding window of recent samples, predict whether a congestion event will start within the next h milliseconds.

Following a Titanic-style walkthrough, we will:

explore the data: distributions, autocorrelation, event durations and class imbalance;
define labels (queue above a threshold for k consecutive samples) and engineer features (lags, moving averages, slopes, rolling variance);
train models of increasing complexity with scikit-learn, LightGBM and PyTorch (logistic regression, gradient-boosted trees, a small GRU) and compare them with simple threshold and moving-average rules;
evaluate with precision, recall, F1 and PR-AUC, splitting train and test by recording to avoid leakage, and measure how much warning time each model gives as h grows.

Early congestion forecasting is a building block for autonomous network operations.
