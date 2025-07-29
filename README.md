# BDT_labEL: Network QoS Analysis & Prediction

## Overview

This project provides a comprehensive toolkit for analyzing, aggregating, and predicting network Quality of Service (QoS) using a combination of Hadoop MapReduce (Java), PySpark, and interactive Python applications (Streamlit and PyQt6). It includes:

- **Data aggregation and feature engineering** using Hadoop MapReduce.
- **Exploratory Data Analysis (EDA)** with Apache Spark.
- **QoS prediction and route optimization** with PySpark and PyQt6 GUIs.
- **Rich visualizations** and result outputs.

---

## Project Structure

```
BDT_labEL/
│
├── artifacts/                # Hadoop/Spark job artifacts
├── output/                   # Aggregated output from MapReduce jobs
│   ├── qosagg/
│   ├── qosagg2/
│   └── qosagg3/
├── data_analysis/            # EDA and local analysis (Python/Streamlit)
│   ├── app.py
│   ├── localanalysis.ipynb
│   ├── sparkanalysis.ipynb
│   └── Train.csv
├── QoS_prediction/           # PySpark + PyQt6 prediction GUIs
│   ├── qos_routing_app.py
│   ├── qos_routing_app_simple.py
│   └── Train.csv
├── Results/                  # Visualizations and analysis results
├── QoSDriver.java            # Hadoop MapReduce driver
├── QoSMapper.java            # Hadoop MapReduce mapper
├── QoSReducer.java           # Hadoop MapReduce reducer
├── qosagg.jar                # Compiled MapReduce job
├── first60.csv               # Sample of the main dataset
└── Train.csv                 # Main dataset (large)
```

---

## Data

- **Train.csv**: The main dataset (large, ~10MB), present in several folders. It contains network measurements, device info, weather, and target QoS values.
- **first60.csv**: A 60-row sample of the main dataset for quick inspection and testing.

**Sample columns:**
```
id, timestamp, device, PCell_RSRP_max, ..., Traffic Jam Factor, area, target
```
(See `first60.csv` for a full header and sample rows.)

---

## Java: Hadoop MapReduce Aggregation

- **QoSDriver.java**: Entry point for running the MapReduce job. Expects input and output paths as arguments.
- **QoSMapper.java**: Extracts and emits key features (RSRP, SNR, speed, congestion, humidity, precipProb) grouped by area and hour.
- **QoSReducer.java**: Aggregates features by averaging them for each area-hour group.

**Usage:**
```sh
hadoop jar qosagg.jar QoSDriver <input_path> <output_path>
```
Output is written to the `output/qosagg*` folders.

---

## Python: Data Analysis & Visualization

### 1. Streamlit EDA App (`data_analysis/app.py`)

- Loads and cleans the dataset.
- Engineers time-based features.
- Visualizes missing values, feature distributions, and correlations.
- Interactive map and plots using Folium, Matplotlib, and Seaborn.

**Run:**
```sh
streamlit run data_analysis/app.py
```

### 2. Jupyter Notebooks

- **localanalysis.ipynb** and **sparkanalysis.ipynb**: Contain in-depth EDA, feature engineering, and Spark-based analysis. (Files are large.)

---

## Python: QoS Prediction & Routing (PySpark + PyQt6)

### 1. `QoS_prediction/qos_routing_app.py`

- Full-featured PyQt6 GUI for loading data, running PySpark-based feature engineering, and optimizing route weights.
- Includes hardware monitoring simulation, advanced plotting, and optimization (grid/random search).
- Uses PySpark for backend processing and Matplotlib for visualization.

### 2. `QoS_prediction/qos_routing_app_simple.py`

- Simpler PyQt6 GUI for running the core Spark pipeline and visualizing results.
- Handles Spark session setup, feature normalization, scoring, and best route selection.

**Run:**
```sh
python QoS_prediction/qos_routing_app.py
# or
python QoS_prediction/qos_routing_app_simple.py
```

---

## Results & Visualizations

- **Results/**: Contains PNGs of feature distributions, correlation matrices, time series, and network performance maps.

---

## Artifacts & Output

- **artifacts/**: Contains Spark/Hadoop job artifacts.
- **output/qosagg*/part-r-00000**: Aggregated results from MapReduce jobs.

---

## Requirements

- **Java** (for Hadoop MapReduce)
- **Python 3.8+**
  - `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `folium`, `PyQt6`, `pyspark`
- **Hadoop** (for running MapReduce jobs)
- **Spark** (for PySpark processing)

---

## Quick Start

1. **Run MapReduce Aggregation:**
   ```sh
   hadoop jar qosagg.jar QoSDriver <input_path> <output_path>
   ```

2. **Explore Data with Streamlit:**
   ```sh
   streamlit run data_analysis/app.py
   ```

3. **Run QoS Prediction GUI:**
   ```sh
   python QoS_prediction/qos_routing_app.py
   ```
