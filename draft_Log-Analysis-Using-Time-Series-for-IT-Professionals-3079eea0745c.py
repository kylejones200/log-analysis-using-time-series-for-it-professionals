# Description: Short example for Log Analysis Using Time Series for IT Professionals.



from statsmodels.tsa.arima.model import ARIMA
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Example log
logs = [
    "2025-01-01 12:00:00, INFO, ResponseTime=200ms",
    "2025-01-01 12:01:00, ERROR, ResponseTime=500ms",
]

# Parsing logs into a DataFrame
parsed_logs = []
for log in logs:
    parts = log.split(", ")
    timestamp, level, metric = parts[0], parts[1], parts[2].split("=")[1]
    parsed_logs.append({"Timestamp": timestamp, "Level": level, "ResponseTime": int(metric[:-2])})

df = pd.DataFrame(parsed_logs)
logger.info(df)

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
aggregated = df.resample('1T').mean()  # Resample data into 1-minute intervals
logger.info(aggregated)

# Example anomaly detection using Z-score

aggregated['Z_Score'] = (aggregated['ResponseTime'] - aggregated['ResponseTime'].mean()) / aggregated['ResponseTime'].std()
anomalies = aggregated[aggregated['Z_Score'].abs() > 2]  # Identify anomalies
logger.info(anomalies)


plt.plot(aggregated.index, aggregated['ResponseTime'], label='Response Time')
plt.title("Response Time Over Time")
plt.xlabel("Time")
plt.ylabel("Response Time (ms)")
plt.legend()
plt.show()


# Fit ARIMA model
model = ARIMA(aggregated['ResponseTime'].dropna(), order=(1, 1, 1))
fitted_model = model.fit()

# Forecast next 10 intervals
forecast = fitted_model.forecast(steps=10)
logger.info(forecast)

# Simulated data for correlation
df['CPU_Usage'] = [50, 70, 60, 80, 55]
correlation = df.corr()
logger.info(correlation)
