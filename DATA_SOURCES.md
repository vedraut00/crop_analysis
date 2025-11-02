# Data Sources and References

## Dataset Information

### Primary Dataset: Synthetic Data

This project uses **synthetically generated data** created specifically for this case study. The data is generated programmatically in `crop_analysis.py`.

**Why Synthetic Data?**
- ✅ **Reproducibility**: Fixed random seed (42) ensures identical results
- ✅ **Controlled Experiments**: Known relationships validate model behavior
- ✅ **Educational Purpose**: Demonstrates complete ML pipeline
- ✅ **No Privacy Issues**: No real farmer or proprietary data
- ✅ **Customizable**: Adjustable parameters for different scenarios

**Generation Details:**
- **File**: `crop_analysis.py`
- **Function**: `generate_agricultural_data(n_samples=200)`
- **Random Seed**: 42 (for reproducibility)
- **Sample Size**: 200 observations
- **Features**: 4 input features + 2 target variables

---

## Similar Real-World Agricultural Datasets

For reference and comparison, here are publicly available agricultural datasets:

### 1. UCI Machine Learning Repository - Crop Recommendation Dataset
**URL**: https://archive.ics.uci.edu/ml/datasets/Crop+recommendation

**Description**: Dataset for recommending crops based on soil and climate conditions
- Features: N, P, K, temperature, humidity, pH, rainfall
- Target: Crop type (22 different crops)
- Samples: 2,200 observations

**Citation**:
```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
School of Information and Computer Science.
```

---

### 2. Kaggle - Crop Yield Prediction Dataset
**URL**: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

**Description**: Historical crop yield data with weather and soil parameters
- Features: Area, Item (crop), Year, rainfall, pesticides, temperature
- Target: Yield (hg/ha)
- Samples: Varies by region

**Usage**: Requires Kaggle account (free)

---

### 3. FAO Agricultural Statistics (FAOSTAT)
**URL**: http://www.fao.org/faostat/en/#data

**Description**: Comprehensive global agricultural statistics from the Food and Agriculture Organization
- **Crop Production**: Production quantities, area harvested, yield
- **Climate Data**: Temperature, precipitation
- **Fertilizer Use**: By country and crop
- **Coverage**: Global, 1961-present

**Datasets Available**:
- Production: http://www.fao.org/faostat/en/#data/QC
- Fertilizers: http://www.fao.org/faostat/en/#data/RFN
- Climate: http://www.fao.org/faostat/en/#data/ET

**Citation**:
```
FAO. (2024). FAOSTAT Statistical Database. 
Food and Agriculture Organization of the United Nations. 
Rome. http://www.fao.org/faostat/
```

---

### 4. USDA National Agricultural Statistics Service (NASS)
**URL**: https://www.nass.usda.gov/

**Quick Stats Database**: https://quickstats.nass.usda.gov/

**Description**: Official US agricultural statistics
- **Crop Production**: Yield, area planted, area harvested
- **Weather Data**: Temperature, precipitation
- **Soil Data**: Moisture, conditions
- **Coverage**: US states and counties, 1866-present

**API Access**: https://quickstats.nass.usda.gov/api

**Citation**:
```
USDA National Agricultural Statistics Service. (2024). 
Quick Stats Database. Washington, DC. 
https://www.nass.usda.gov/
```

---

### 5. NASA POWER Project - Agroclimatology Data
**URL**: https://power.larc.nasa.gov/

**Data Access Viewer**: https://power.larc.nasa.gov/data-access-viewer/

**Description**: NASA's Prediction of Worldwide Energy Resources
- **Solar Radiation**: Daily, monthly averages
- **Temperature**: Min, max, average
- **Precipitation**: Daily rainfall
- **Humidity**: Relative humidity
- **Coverage**: Global, 1981-present

**API Documentation**: https://power.larc.nasa.gov/docs/

**Citation**:
```
NASA Langley Research Center (LaRC) POWER Project. (2024). 
Prediction Of Worldwide Energy Resources. 
https://power.larc.nasa.gov/
```

---

### 6. World Bank - Climate Change Knowledge Portal
**URL**: https://climateknowledgeportal.worldbank.org/

**Description**: Climate and agricultural data by country
- Historical climate data
- Agricultural statistics
- Crop suitability maps
- Climate projections

---

### 7. CGIAR - Agricultural Research Data
**URL**: https://www.cgiar.org/

**Dataverse**: https://dataverse.harvard.edu/dataverse/CGIAR

**Description**: International agricultural research data
- Crop trials
- Soil data
- Climate data
- Genetic resources

---

### 8. European Commission - MARS Crop Yield Forecasting
**URL**: https://ec.europa.eu/jrc/en/mars

**Description**: Monitoring Agricultural Resources (MARS)
- Crop yield forecasts
- Weather data
- Soil moisture
- Coverage: Europe and global

---

### 9. India - Agricultural Statistics
**URL**: https://eands.dacnet.nic.in/

**Description**: Directorate of Economics and Statistics
- Crop production
- Area and yield
- State-wise data
- Coverage: India, 1950-present

---

### 10. Australia - ABARES Agricultural Data
**URL**: https://www.agriculture.gov.au/abares

**Description**: Australian Bureau of Agricultural and Resource Economics and Sciences
- Crop production
- Farm surveys
- Climate data
- Coverage: Australia

---

## Academic Datasets

### 11. Crop Yield Prediction - Research Papers

**Paper**: Khaki, S., & Wang, L. (2019). "Crop Yield Prediction Using Deep Neural Networks."
**Dataset**: Available upon request from authors
**URL**: https://www.frontiersin.org/articles/10.3389/fpls.2019.00621/full

**Paper**: Crane-Droesch, A. (2018). "Machine learning methods for crop yield prediction"
**Dataset**: USDA NASS data
**URL**: https://iopscience.iop.org/article/10.1088/1748-9326/aae159

---

## How to Use These Datasets

### For This Project:
The synthetic data in this project is sufficient for demonstrating ML techniques and completing the assignment.

### For Extended Research:
1. **Download** data from any of the above sources
2. **Preprocess** to match the format in `agricultural_data.csv`
3. **Replace** the data generation in `crop_analysis.py`
4. **Run** the same analysis pipeline

### Example Code to Load Real Data:
```python
# Instead of generating synthetic data
# df = generate_agricultural_data(200)

# Load real data
df = pd.read_csv('real_agricultural_data.csv')

# Ensure columns match expected format:
# soil_moisture, rainfall, temperature, fertilizer, crop_type, yield
```

---

## Data Citation for This Project

When citing the data used in this project:

**Synthetic Data:**
```
Dataset: Synthetically generated agricultural data
Source: Custom generation script (crop_analysis.py)
Date: November 2, 2025
Samples: 200 observations
Features: soil_moisture, rainfall, temperature, fertilizer
Targets: crop_type (binary), yield (continuous)
Reproducibility: Random seed = 42
```

**Similar Real Datasets (for reference):**
```
1. UCI Machine Learning Repository - Crop Recommendation Dataset
   https://archive.ics.uci.edu/ml/datasets/Crop+recommendation

2. Kaggle - Crop Yield Prediction Dataset
   https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

3. FAO FAOSTAT - Agricultural Statistics
   http://www.fao.org/faostat/en/#data

4. USDA NASS - Quick Stats Database
   https://www.nass.usda.gov/

5. NASA POWER - Agroclimatology Data
   https://power.larc.nasa.gov/
```

---

## Additional Resources

### Agricultural ML Research:
- **Review Paper**: Liakos et al. (2018). "Machine Learning in Agriculture: A Review"
  - URL: https://www.mdpi.com/1424-8220/18/8/2674
  - Comprehensive overview of ML applications in agriculture

### Python Libraries for Agricultural Data:
- **PyAEZ**: Agro-Ecological Zoning framework
  - URL: https://github.com/gicait/PyAEZ
  
- **CropSyst**: Crop simulation system
  - URL: http://modeling.bsyse.wsu.edu/CS_Suite/CropSyst/

### Online Courses:
- **Coursera**: "Data Science for Agriculture"
- **edX**: "Sustainable Food Security"

---

## Summary

This project uses **synthetic data** for educational purposes, but the methodology and code can be applied to any of the real-world datasets listed above. The synthetic data approach ensures:

✅ Reproducibility across all users
✅ Controlled experimental conditions
✅ No data access or privacy issues
✅ Fast execution and testing
✅ Clear demonstration of ML concepts

For production applications or research publications, consider using the real-world datasets referenced above.

---

**Last Updated**: November 2, 2025
**Version**: 1.0
