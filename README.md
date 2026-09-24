# SafeHer-AI

## Geo-Fenced Emergency Alert and Safety Assistance System for Women

SafeHer-AI is a technology-driven women safety system designed to provide **location-aware safety assistance, emergency alerts, risk analysis, and GIS-based visualization of potentially unsafe areas**.

The project combines **Web Technologies, Machine Learning/Deep Learning, GIS, and Cloud Computing** to analyze historical crime data, identify area-level risk patterns, and provide a platform for safety assistance.

---

## 📌 Project Overview

Women safety in urban and semi-urban areas can be improved through systems that combine historical crime information with geographic location and intelligent risk analysis.

SafeHer-AI aims to develop a system that:

* Analyzes historical crime data.
* Performs data preprocessing and feature engineering.
* Uses machine learning models for risk analysis.
* Classifies geographical areas into different risk levels.
* Uses GIS to visualize risk geographically.
* Implements geofencing based on identified risk areas.
* Provides emergency assistance through a web interface.
* Generates location-aware safety information.

The project is developed as a **final-year mini project** under the domain of **Artificial Intelligence and Data Science**.

---

## 🎯 Objectives

The main objectives of SafeHer-AI are:

1. To analyze historical crime data related to women safety.
2. To preprocess and transform crime datasets into usable machine-learning features.
3. To identify patterns in crime occurrence across geographical regions.
4. To develop machine-learning models for area-level risk analysis.
5. To categorize areas into different risk bands.
6. To integrate GIS for geographic visualization.
7. To implement geofencing around identified risk areas.
8. To provide emergency alert and safety assistance functionality.
9. To develop a web-based interface for interacting with the system.
10. To contribute towards **SDG 5 – Gender Equality** through technology-enabled safety assistance.

---

## 🧠 Technologies Used

| Category              | Technologies                            |
| --------------------- | --------------------------------------- |
| Programming           | Python, JavaScript                      |
| Frontend              | HTML5, CSS3, JavaScript                 |
| Backend               | Python, FastAPI                         |
| Machine Learning      | Scikit-learn                            |
| Data Processing       | Pandas, NumPy                           |
| GIS                   | GeoJSON, GIS-based mapping              |
| Database/Data Storage | CSV datasets                            |
| Visualization         | HTML/JavaScript-based GIS visualization |
| Development           | VS Code, Git, GitHub                    |

---

## 🤖 Machine Learning

The current implementation includes machine-learning based risk analysis.

### MLP – Multi-Layer Perceptron

An **MLP neural network** is used for risk analysis based on engineered crime-related features.

The project contains different MLP training implementations, including:

* Dataset 01 MLP
* Bounded MLP
* Combined Dataset 01/02/03 MLP
* MLP evaluation scripts

The trained models and associated preprocessing outputs are stored inside the `models/` directory.

### K-Means Clustering

K-Means clustering is also included for exploratory risk-band analysis.

It is used to group areas based on similarities in their crime-related characteristics.

---

## 🗺️ GIS and Geo-Fencing

GIS is an important component of SafeHer-AI.

The system processes geographical boundaries and risk information to create a **risk layer** that can be visualized on a map.

The GIS implementation contains:

```text
gis/
├── data/
├── geofencing/
├── map/
├── outputs/
└── scripts/
```

The `geofencing/` module contains the risk geofencing implementation.

The system generates geographical risk information that can be used to identify areas associated with different risk levels.

---

## 📊 Dataset and Data Processing

The project currently works with historical crime datasets covering multiple years.

The data-processing pipeline includes:

```text
Raw Crime Data
       ↓
Data Cleaning
       ↓
Data Harmonization
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Risk Analysis
       ↓
Machine Learning
       ↓
Risk Bands
       ↓
GIS Risk Layer
       ↓
Geo-Fenced Safety Visualization
```

The processed datasets are maintained inside the:

```text
data/
```

directory.

---

## 🏗️ Project Architecture

The overall system follows this workflow:

```text
                ┌─────────────────────┐
                │   Historical Crime  │
                │       Dataset       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Data Preprocessing   │
                │ & Cleaning           │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Engineering │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ ML Risk Analysis    │
                │ MLP / K-Means       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Risk Band           │
                │ Classification       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ GIS Risk Layer      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Geo-Fencing         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ SafeHer-AI Web      │
                │ Application         │
                └─────────────────────┘
```

---

## 📁 Project Structure

```text
SafeHer-AI/
│
├── alerts/
│
├── app/
│   ├── main.py
│   ├── test_dataset_01.py
│   │
│   ├── frontend/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   │
│   └── routers/
│       ├── risk.py
│       └── __init__.py
│
├── data/
│   ├── cleaned_district_ipc_2001_2012.csv
│   ├── cleaned_district_ipc_2001_2012_v2.csv
│   ├── dstrIPC_1.csv
│   ├── dstrIPC_1_2014.csv
│   ├── dstrIPC_2013.csv
│   ├── dataset_02_03_2013_2014/
│   ├── eda_dataset_01/
│   ├── feature_engineering_01/
│   └── master_2001_2014/
│
├── gis/
│   ├── data/
│   ├── geofencing/
│   │   └── risk_geofence.py
│   ├── map/
│   ├── outputs/
│   └── scripts/
│
├── models/
│   ├── train_mlp_01.py
│   ├── train_bounded_mlp_01.py
│   ├── train_mlp_2001_2014.py
│   ├── evaluate_risk_bands_2001_2014.py
│   ├── dataset_01_mlp/
│   ├── dataset_01_bounded_mlp/
│   ├── dataset_01_02_03_mlp/
│   ├── dataset_01_risk_bands/
│   ├── dataset_01_02_03_risk_bands/
│   └── dataset_02_03_kmeans/
│
├── preprocessing/
│   └── preprocessing scripts
│
├── notebooks/
│
├── .gitignore
└── README.md
```

---

## ⚙️ Backend

The backend is implemented using **FastAPI**.

The main application is:

```text
app/main.py
```

The risk-related API functionality is organized under:

```text
app/routers/risk.py
```

The backend is responsible for connecting the frontend with the risk-analysis functionality.

---

## 🌐 Frontend

The frontend is implemented using:

* HTML5
* CSS3
* JavaScript

Frontend files:

```text
app/frontend/
├── index.html
├── app.js
└── style.css
```

The frontend provides the interface through which users can interact with the SafeHer-AI system.

---

## 🚨 Emergency Assistance

The system is designed to support emergency safety assistance through its application layer.

The intended workflow is:

```text
User Location
     ↓
Location/Risk Analysis
     ↓
Risk Area Detection
     ↓
Geo-Fence Evaluation
     ↓
Safety Assistance / Alert
```

The `alerts/` component is reserved for the emergency-alert functionality of the system.

---

## 📍 GIS Risk Visualization

The project generates GIS-compatible risk layers.

The generated outputs include:

* Risk-related CSV files
* GIS output files
* GeoJSON risk layers

These outputs allow risk information to be associated with geographical regions and displayed through the map interface.

---

## 📈 Current Implementation Status

### Completed / Implemented

* [x] Project structure created
* [x] Historical crime datasets collected and organized
* [x] Data cleaning
* [x] Dataset harmonization
* [x] Exploratory Data Analysis
* [x] Feature engineering
* [x] MLP model implementations
* [x] Bounded MLP implementation
* [x] K-Means risk analysis
* [x] Risk-band generation
* [x] GIS processing
* [x] GeoJSON risk-layer generation
* [x] Geofencing module
* [x] FastAPI backend structure
* [x] Frontend structure
* [x] Git/GitHub project integration

### Under Development

* [ ] Complete emergency alert workflow
* [ ] Complete frontend-backend integration
* [ ] Final geo-fence interaction
* [ ] Final user-location integration
* [ ] End-to-end testing
* [ ] Final model evaluation
* [ ] Final deployment

---

## 🔮 Expected Final Output

The final SafeHer-AI system is expected to provide a web-based safety assistance platform where:

1. The user interacts with the SafeHer-AI web application.
2. The user's geographical location can be evaluated.
3. The system determines the corresponding geographical risk information.
4. GIS data is used to identify risk zones.
5. Geo-fencing can identify whether the user enters a defined risk area.
6. The system provides appropriate safety assistance.
7. Emergency functionality can be triggered when required.
8. Risk information can be visualized through an interactive map.

---

## 🌱 Sustainable Development Goal

### SDG 5 – Gender Equality

The project is aligned with **Sustainable Development Goal 5**, particularly through the development of technology-enabled mechanisms intended to support women's safety and accessibility to safety assistance.

---

## 👨‍💻 Team Members

### Kushal K S

**USN:** 20231IST0025

### Adesh Mathpati

**USN:** 20231IST0060

### Yashas N P
**USN:** 20231IST0053

**Department:** Information Science and Technology – Artificial Intelligence & Data Science

---

## 🔬 Project Domain

```text
Artificial Intelligence
        +
Machine Learning
        +
Deep Learning
        +
GIS
        +
Web Technologies
        +
Cloud Computing
```

---

## 📌 Project Status

**SafeHer-AI is currently under development as a final-year mini project.**

The repository contains the current implementation, datasets, preprocessing modules, machine-learning models, GIS processing components, backend, frontend, and generated outputs.






