# 🌫️ Air Pollution Analysis in India - 2024

## 📘 Project Description

Air pollution in India is a severe environmental issue, with the country ranking among the highest in global pollution levels. According to the **World Air Quality Report (2023)** by IQAir, **26 of the 30 most polluted cities in the world are in India**, with **Delhi ranked third** globally.

Delhi recorded its **worst air pollution in five years** during **December 2023**, as reported by the Ministry of Environment, Forest and Climate Change. The primary contributors to this crisis include:

- Vehicular emissions
- Industrial pollution
- Seasonal stubble burning (especially during winter)

This project focuses on analyzing the air quality data and building models to study and forecast pollution patterns.

---

## 📊 Dataset

The dataset is sourced from the **Central Pollution Control Board (CPCB)** and includes daily air quality and meteorological data from **1st January 2024 to 31st December 2024**.

### Included Parameters:

- **Air Pollutants**: PM2.5, PM10, NO, NO₂, NOx, NH₃, SO₂, CO, O₃, Benzene (C₆H₆), Toluene (C₇H₈)
- **Weather Data**: Relative Humidity (RH), Wind Speed (WS), Wind Direction (WD), Solar Radiation (SR), Barometric Pressure (BP), Ambient Temperature (AT), Rainfall (RF), Total Rainfall (TOT-RF)

---

## 🗂️ File Structure

project/
├── trainingpipeline.py
├── SARIMAX.py
├── requirements.txt
├── README.md
└── data/
└── air_quality_2024.csv


---

## 🛠️ Setup Instructions

Follow these steps to set up and run the project:

### 1. Open in VS Code

If you have VS Code installed:

'bash
code .'

### 2. Check for versions of python and anaconda.
python --version
conda --version

### 3. Create and Activate Environment
conda create --name venv python==3.8 -y
conda activate venv

### 4. Install Dependencies

pip install -r requirements.txt


### 5.Run the Training Pipeline

python trainingpipeline.py/(path of training pipeline file)

### 6.Run Time Series Model
python SARIMAX.py/(path of SARIMAX file)