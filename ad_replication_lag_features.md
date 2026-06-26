# Feature Engineering Mathematics for AD Replication Lag

## 1. Core Data Sources & Raw Metrics

### Primary Inputs
- **`lastSyncSuccessTime`**: Timestamp of last successful replication per partner.
- **`usnLastObjectApplied`**: Highest Update Sequence Number (USN) applied from a partner.
- **`usnSource`**: Current highest USN on the source DC.
- **`syncLatency`**: Time taken for the last sync operation (ms).
- **`failureCount`**: Consecutive failed replication attempts.
- **`partitionSize`**: Number of objects in the partition (NTDS.DIT size context).

---

## 2. Mathematical Formulations

### A. Temporal Lag Features (Time-Domain)

#### 1. Absolute Time Lag ($L_{time}$)
The raw difference between current time and last successful sync.
$$ L_{time}(t) = t_{now} - t_{last\_success} $$
*Unit: Seconds*

#### 2. Exponential Moving Average (EMA) of Lag
Smooths noise to identify trend direction vs. transient spikes.
$$ EMA_{lag}(t) = \alpha \cdot L_{time}(t) + (1 - \alpha) \cdot EMA_{lag}(t-1) $$
- $\alpha$ (Smoothing Factor): Typically $0.1$ to $0.3$. Lower $\alpha$ = smoother trend.
- *Use Case*: Detecting slow drifts in replication health.

#### 3. Lag Rate of Change (Velocity & Acceleration)
First derivative (Velocity): How fast is the lag growing?
$$ v(t) = \frac{L_{time}(t) - L_{time}(t-\Delta t)}{\Delta t} $$

Second derivative (Acceleration): Is the problem worsening exponentially?
$$ a(t) = \frac{v(t) - v(t-\Delta t)}{\Delta t} $$
- *Interpretation*: $a(t) > 0$ indicates a cascading failure or network saturation.

---

### B. Sequence Lag Features (USN-Domain)

#### 4. USN Delta Gap
Measures the volume of missed changes rather than just time.
$$ G_{usn}(t) = USN_{source\_max} - USN_{applied\_from\_partner} $$
- *Normalization*: Convert to "Expected Sync Time" based on historical write throughput.
$$ T_{expected} = \frac{G_{usn}(t)}{R_{write\_avg}} $$
Where $R_{write\_avg}$ is the average writes/second on the source DC.

#### 5. USN Stagnation Index
Binary or ratio feature indicating if USN has not moved despite time passing.
$$ I_{stagnation} = \begin{cases} 
1 & \text{if } (USN(t) - USN(t-\Delta t)) = 0 \land (t_{now} - t_{last\_attempt}) > \theta \\
0 & \text{otherwise}
\end{cases} $$

---

### C. Statistical & Distributional Features

#### 6. Z-Score of Latency
Identifies outliers against the partner's historical baseline.
$$ Z(t) = \frac{SyncLatency(t) - \mu_{partner}}{\sigma_{partner}} $$
- $\mu_{partner}$: Rolling mean latency for this specific source-destination pair.
- $\sigma_{partner}$: Rolling standard deviation.
- *Threshold*: $|Z| > 3$ usually indicates an anomaly.

#### 7. Coefficient of Variation (CV)
Measures consistency of replication intervals. High CV implies unstable network or DC load.
$$ CV = \frac{\sigma_{intervals}}{\mu_{intervals}} $$

#### 8. Entropy of Failure Codes
If multiple error codes exist (e.g., RPC unavailable, Access Denied), calculate entropy to detect chaotic failure modes vs. consistent single errors.
$$ H(X) = -\sum P(x_i) \log_2 P(x_i) $$
- Low Entropy: Consistent single issue (easier to fix).
- High Entropy: Systemic instability.

---

### D. Topological & Contextual Features

#### 9. Partner Weighted Lag
Not all partners are equal. Weight lag by the criticality of the source (e.g., PDC Emulator).
$$ L_{weighted} = \sum_{i=1}^{n} (w_i \cdot L_{time, i}) $$
- $w_i$: Criticality score (1.0 for PDCe, 0.5 for Hub, 0.2 for RoDC).

#### 10. Network Hop Penalty
Incorporate site link costs into the expected lag baseline.
$$ L_{adjusted} = L_{time} - (Cost_{site\_link} \times K_{network\_factor}) $$
- If $L_{adjusted} \gg 0$, the lag is due to local DC issues, not network design.

#### 11. Time-of-Day Seasonality Residuals
Remove expected daily patterns (e.g., high lag during backup windows).
$$ Residual(t) = L_{time}(t) - Forecast_{seasonal}(t) $$
- Use Fourier Transform or STL decomposition to extract $Forecast_{seasonal}$.

---

## 3. Advanced Derived Features for ML Models

### A. "Time to Failure" Proxy (Survival Analysis)
Estimate time until the USN gap exceeds the tombstone lifetime (default 180 days) or buffer limits.
$$ T_{critical} = \frac{Limit_{max\_usn\_gap} - G_{usn}(t)}{v(t) + \epsilon} $$
- If $v(t) \le 0$, $T_{critical} = \infty$.
- *Usage*: Regression target for predicting imminent forest trust breaks.

### B. Interaction Terms
Capture non-linear relationships between load and lag.
$$ Feature_{interaction} = L_{time} \times CPU_{usage} \times DiskQueueLength $$
- Helps models learn that lag is only critical when accompanied by resource contention.

### C. Rolling Window Aggregates
Compute over windows $W = [1h, 4h, 24h, 7d]$:
- $\max(L_{time})_W$: Peak stress.
- $\min(L_{time})_W$: Best case baseline.
- $Slope(L_{time})_W$: Linear regression slope over the window.

---

## 4. Implementation Logic (Pseudocode)

```python
def compute_replication_features(current_row, history_window):
    # 1. Basic Lag
    lag_time = (now - current_row.last_sync_success).total_seconds()
    
    # 2. EMA Lag
    alpha = 0.2
    ema_lag = alpha * lag_time + (1 - alpha) * history_window['ema_lag'].iloc[-1]
    
    # 3. Velocity (Rate of Change)
    if len(history_window) > 1:
        prev_lag = history_window['lag_time'].iloc[-1]
        delta_t = (current_row.timestamp - history_window['timestamp'].iloc[-1]).total_seconds()
        velocity = (lag_time - prev_lag) / delta_t if delta_t > 0 else 0
    else:
        velocity = 0
        
    # 4. USN Gap Normalized by Write Rate
    usn_gap = current_row.source_usn_max - current_row.applied_usn
    avg_write_rate = history_window['writes_per_sec'].mean()
    expected_catchup_time = usn_gap / avg_write_rate if avg_write_rate > 0 else float('inf')
    
    # 5. Z-Score
    mean_latency = history_window['sync_latency'].mean()
    std_latency = history_window['sync_latency'].std()
    z_score = (current_row.sync_latency - mean_latency) / std_latency if std_latency > 0 else 0
    
    return {
        'lag_time': lag_time,
        'ema_lag': ema_lag,
        'lag_velocity': velocity,
        'expected_catchup_time': expected_catchup_time,
        'latency_z_score': z_score,
        'is_stagnant': 1 if (usn_gap > 0 and velocity == 0) else 0
    }
```

## 5. Validation Strategy
1. **Stationarity Check**: Use Augmented Dickey-Fuller (ADF) test on $L_{time}$ series. Differencing may be required before feeding into ARIMA/LSTM.
2. **Correlation Matrix**: Ensure `lag_time` and `usn_gap` aren't perfectly collinear; if so, keep the one with higher predictive power for the specific failure mode.
3. **Feature Importance**: Train a Random Forest on historical outage data to rank these engineered features. Drop those with near-zero importance.
