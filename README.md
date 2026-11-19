# AI-Project-G17

Task 1: group members
- Edouard de Noblet, Mechanical Engineering, edenoblet@gmail.com
- Mohamed Ali Kabiri, Computer Science, mohamedali.kabiri@gmail.com
- Lorenzo Soubiran Frixione, Computer Science, lorenzo.soubiranfrixione@edu.ece.fr
- Zineb El Koussaimi, Computer science, koussaimizineb@gmail.com

Research idea

Title: Multi-horizon Forecasting of Switch Queue Occupancy from High-Frequency Telemetry, with Prediction Intervals

Proposal (Option A): A switch queue can fill within a few milliseconds when competing traffic turns on, which causes delay spikes and packet loss. Reacting only once the queue is full is too late, so we will forecast how full it will be and how uncertain that forecast is. The dataset is a set of already-collected, high-frequency (500 Hz) queue traces from an emulated network testbed (BMv2 software switches in Mininet). Every 2 ms, a measurement packet reports the depth and delay of the queue it leaves, at the bottleneck switch and at a downstream switch. The task: given the telemetry up to sample t, forecast the bottleneck queue depth at t+1, t+5, t+10 and t+25 samples (2, 10, 20 and 50 ms ahead), as a point forecast and as the 0.1, 0.5 and 0.9 quantiles.

Following a Titanic-style walkthrough, we will:

explore the data: distributions, autocorrelation, how predictability decays with the horizon, and lost samples; build supervised windows and engineer features (lags, moving averages, slopes, rolling variance, the downstream switch's readings);train models of increasing complexity (a statistical model, gradient-boosted trees, a recurrent network, an attention or TCN model) and compare them with persistence and EWMA baselines; evaluate each horizon with MAE and RMSE, pinball loss for the0.1/0.5/0.9 quantiles and the coverage of the 80% interval, splitting train, validation and test by recording to avoid leakage, and using the test recordings only once, at the end.

Forecasting queue occupancy with calibrated uncertainty is a building block for autonomous network operations.
