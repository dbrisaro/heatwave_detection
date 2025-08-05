# Heatwave Detection

Identification and analysis of heatwave events using ERA5 data for a determined region

## Structure

- `notebooks/`: Data analysis and processing
  - ERA5 data download
  - Percentile computation
  - Heatwave detection
  - Analysis and comparison with station data

## Data

- Daily maximum temperature (ERA5)
- Daily total precipitation (ERA5)
- Period: 1960-2025
- Region: 5°N-45°N, 95°W-65°W

## Methodology

Heatwave detection using:
- Percentile method (p90, p95)
- Climate anomaly method
- Analysis at specific points and on grid