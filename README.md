# xLSTF
Efficient Long Term Time Series Forecasting (LTSF)

# Multivariate Datasets

This project utilizes a comprehensive collection of well-established multivariate datasets tailored for long-term time series forecasting tasks. Specifically, it includes 24 multivariate time series datasets from the [Time Series Forecasting Benchmark (TFB)](https://dl.acm.org/doi/abs/10.14778/3665844.3665863).



## Datasets Overview

Below is an overview of the datasets used, along with their key characteristics:

| Dataset Name     | Domain       | Granularity | # Variates | # Time Steps | Look-Back Window | Forecasting Horizon Lengths | Split Ratio (Train/Val/Test) |
|------------------|--------------|-------------|------------|--------------|------------------------|-----------------------------|------------------------------|
| **METR-LA**      | Traffic      | 5 minutes   | 207        | 34,272       | 336                    | (96, 192, 336, 720)         | (0.7, 0.1, 0.2)              |
| **PEMS-BAY**     | Traffic      | 5 minutes   | 325        | 52,116       | 336                    | (96, 192, 336, 720)         | (0.7, 0.1, 0.2)              |
| **PEMS08**       | Traffic      | 5 minutes   | 170        | 17,856       | 336                    | (96, 192, 336, 720)         | (0.6, 0.2, 0.2)              |
| **Traffic**      | Traffic      | 1 hour      | 862        | 17,544       | 336                    | (96, 192, 336, 720)         | (0.7, 0.1, 0.2)              |
| **ETTh1**        | Electricity  | 1 hour      | 7          | 14,400       | 336                    | (96, 192, 336, 720)         | (0.6, 0.2, 0.2)              |
| **ETTh2**        | Electricity  | 1 hour      | 7          | 14,400       | 336                    | (96, 192, 336, 720)         | (0.6, 0.2, 0.2)              |
| **ETTm1**        | Electricity  | 15 minutes  | 7          | 57,600       | 336                    | (96, 192, 336, 720)         | (0.6, 0.2, 0.2)              |
| **ETTm2**        | Electricity  | 15 minutes  | 7          | 57,600       | 336                    | (96, 192, 336, 720)         | (0.6, 0.2, 0.2)              |
| **Electricity**  | Electricity  | 1 hour      | 321        | 26,304       | 336                    | (96, 192, 336, 720)         | (0.7, 0.1, 0.2)              |
| **Solar**        | Energy       | 10 minutes  | 137        | 52,560       | 336                    | (96, 192, 336, 720)         | (0.6, 0.2, 0.2)              |
| **Wind**         | Energy       | 15 minutes  | 7          | 48,673       | 336                    | (96, 192, 336, 720)         | (0.7, 0.1, 0.2)              |
| **Weather**      | Environment  | 10 minutes  | 21         | 52,696       | 336                    | (96, 192, 336, 720)         | (0.7, 0.1, 0.2)              |
| **AQShunyi**     | Environment  | 1 hour      | 11         | 35,064       | 336                    | (96, 192, 336, 720)         | (0.6, 0.2, 0.2)              |
| **AQWan**        | Environment  | 1 hour      | 11         | 35,064       | 336                    | (96, 192, 336, 720)         | (0.6, 0.2, 0.2)              |
| **ZafNoo**       | Nature       | 30 minutes  | 11         | 19,225       | 336                    | (96, 192, 336, 720)         | (0.7, 0.1, 0.2)              |
| **CzeLAn**       | Nature       | 30 minutes  | 11         | 19,934       | 336                    | (96, 192, 336, 720)         | (0.7, 0.1, 0.2)              |
| **FRED-MD**      | Economic     | 1 month     | 107        | 728          | 104                    | (24, 36, 48, 60)            | (0.7, 0.1, 0.2)              |
| **Exchange Rate**| Economic     | 1 day       | 8          | 7,588        | 104                    | (24, 36, 48, 60)            | (0.7, 0.1, 0.2)              |
| **NASDAQ**       | Stock        | 1 day       | 5          | 1,244        | 104                    | (24, 36, 48, 60)            | (0.7, 0.1, 0.2)              |
| **NYSE**         | Stock        | 1 day       | 5          | 1,243        | 104                    | (24, 36, 48, 60)            | (0.7, 0.1, 0.2)              |
| **NN5**          | Banking      | 1 day       | 111        | 791          | 104                    | (24, 36, 48, 60)            | (0.7, 0.1, 0.2)              |
| **ILI**          | Health       | 1 week      | 7          | 966          | 104                    | (24, 36, 48, 60)            | (0.7, 0.1, 0.2)              |
| **Covid-19**     | Health       | 1 day       | 948          | 1,392        | 104                    | (24, 36, 48, 60)            | (0.7, 0.1, 0.2)              |
| **Wike2000**     | Web          | 1 day       | 2,000        | 792          | 104                    | (24, 36, 48, 60)            | (0.7, 0.1, 0.2)              |

---
## Datasets Description

### METR-LA Dataset
The [METR-LA Dataset](https://arxiv.org/abs/1707.01926) provides traffic speed data collected from 207 loop detection sensors on the Los Angeles County highway. The dataset spans four months, from March 1, 2012, to June 30, 2012, with a total of 34,272 time steps. Each time step represents a 5-minute interval, capturing the traffic speed changes throughout the day.

### PEMS-BAY Dataset
The [PEMS-BAY Dataset](https://arxiv.org/abs/1707.01926) contains traffic information collected by the Californian Transportation Agencies (CalTrans) Performance Measurement System (PEMS). It includes six months of data, from January 1, 2017, to May 31, 2017, recorded by 325 sensors in the Bay Area. The dataset has a total of 52,116 time steps, with each time step representing a 5-minute interval.

### PEMS08 Dataset
The [PEMS08 Dataset](https://ojs.aaai.org/index.php/AAAI/article/view/5438) consists of traffic data collected by the CalTrans Performance Measurement System (PEMS) using 170 sensors in California. The data was recorded over two months, from July 1, 2016, to August 31, 2016, resulting in a total of 17,856 time steps. Each time step represents a 5-minute interval.

### Traffic Dataset
The [Traffic Dataset](https://pems.dot.ca.gov/) includes traffic occupancy rates information based on CalTrans' Performance Measurement System (PEMS) on scale from 0 to 1. It comprises data collected from 862 sensors across California San Francisco freeways over a 24-month period, spanning from 2015 to 2016. The dataset contains a total of 17,544 time steps, each recorded at an hourly interval.

### Electricity Transformer Temperature (ETT) Benchmark
The [ETT Benchmark](https://arxiv.org/pdf/2012.07436) was first introduced in the Informer paper and includes four datasets: ETTh1, ETTh2, ETTm1, and ETTm2. These datasets were obtained from two transformers in China over a 24-month period from July 2016 to July 2018. 
- **ETTh1** and **ETTh2** have a temporal granularity of 1 hour, each containing 14,400 time steps.
- **ETTm1** and **ETTm2** have a finer temporal resolution of 15 minutes, with 57,600 time steps each.
- ETTh1 and ETTm1 were collected from the first transformer, while ETTh2 and ETTm2 were obtained from the second transformer.

| Variable | Description         |
| -------- | ------------------- |
| HUFL     | High Useful Load    |
| HULL     | High Useless Load   |
| MUFL     | Middle Useful Load  |
| MULL     | Middle Useless Load |
| LUFL     | Low Useful Load     |
| LULL     | Low Useless Load    |
| OT       | Oil Temperature     |

### Electricity Dataset
The [Electricity Dataset](https://github.com/laiguokun/multivariate-time-series-data/tree/master) consists of electricity consumption measurements (in kWh) from 321 households, recorded over a two-year period from 2012 to 2014. It contains a total of 26,304 time steps, each representing a 1-hour interval. The dataset is a cleaned version derived from the original [electricity load diagrams](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).

### Solar Dataset
The [Solar Dataset](http://www.nrel.gov/grid/solar-power-data.html) comprises synthetic solar photovoltaic (PV) power plant production records from 137 solar power plants in Alabama, collected in the year 2006. It includes a total of 52,560 time steps with a temporal granularity of 10 minutes. 

### Wind Dataset
The [Wind Dataset](https://proceedings.neurips.cc/paper_files/paper/2022/file/91a85f3fb8f570e6be52b333b5ab017a-Paper-Conference.pdf) consists of six meteorological features, such as predicted wind speed, direction, and temperature. The primary target variable is the electricity production of a wind turbine. It includes 38,673 time steps, recorded every 15 minutes, spanning from January 1, 2020, to May 22, 2021. More information can be found on the [PaddleSpatial GitHub repository](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/paddlespatial/datasets/WindPower).

### Weather Dataset
The [Weather Dataset](https://www.bgc-jena.mpg.de/wetter/) records 21 meteorological indicators throughout the year 2020 with a temporal granularity of 10 minutes. It contains a total of 52,696 time steps. 

| Symbol   | Unit           | Variable                                    |
| -------- | -------------- | ------------------------------------------- |
| p        | mbar           | air pressure                                |
| T        | °C             | air temperature                             |
| T_pot    | K              | potential temperature                       |
| T_dew    | °C             | dew point temperature                       |
| rh       | %              | relative humidity                           |
| VP_max   | mbar           | saturation water vapor pressure             |
| VP_act   | mbar           | actual water vapor pressure                 |
| VP_def   | mbar           | water vapor pressure deficit                |
| sh       | g kg^-1        | specific humidity                           |
| H2OC     | mmol mol^-1    | water vapor concentration                   |
| rho      | g m^-3         | air density                                 |
| wv       | m s^-1         | wind velocity                               |
| max. wv  | m s^-1         | maximum wind velocity                       |
| wd       | degrees        | wind direction                              |
| rain     | mm             | precipitation                               |
| raining  | s              | duration of precipitation                   |
| SWDR     | W m^-2         | short wave downward radiation               |
| PAR      | μmol m^-2 s^-1 | photosynthetically active radiation         |
| max. PAR | μmol m^-2 s^-1 | maximum photosynthetically active radiation |
| Tlog     | °C             | internal logger temperature                 |
| CO2      | ppm            | CO$_2$-concentration of ambient air         |

### AQShunyi and AQWan Datasets
The AQShunyi and AQWan datasets feature 11 air quality measurements from one monitoring station each, collected over four years. Each dataset includes 35,064 time steps with a temporal granularity of 1 hour. More details can be found on [PubMed](https://pubmed.ncbi.nlm.nih.gov/28989318/).

### ZafNoo and CzeLan Datasets
The ZafNoo and CzeLan datasets are based on environmental measurements, each including 11 features. The ZafNoo dataset consists of 19,225 time steps, while the CzeLan dataset has 19,934 time steps, both recorded every 30 minutes from May 6, 2016, to June 25, 2017. Additional information is available on [Tree Physiology](https://academic.oup.com/treephys/article/36/12/1449/2571314).

### FRED-MD Dataset
The [FRED-MD Dataset](https://www.tandfonline.com/doi/full/10.1080/07350015.2015.1086655) collects 107 macroeconomic indicators published by the Federal Reserve Bank. It comprises 748 time steps with a temporal granularity of 1 month.

### Exchange Rate Dataset
The [Exchange Rate Dataset](https://dl.acm.org/doi/abs/10.1145/3209978.3210006) provides daily exchange rates from USD to eight other currencies, including AUD, GBP, CAD, CHF, CNY, JPY, NZD, and SGD. It spans from 1990 to 2016, with a total of 7,588 time steps recorded daily.

| Variate | Country       | Currency                 |
| ------- | ------------- | ------------------------ |
| 0       | Australia     | Australian dollar (AUD)  |
| 1       | Great Britain | Sterling (GBP)           |
| 2       | Canada        | Canadian dollar (CAD)    |
| 3       | Switzerland   | Swiss franc (CHF)        |
| 4       | China         | Renminbi (CNY)           |
| 5       | Japan         | Japanese yen (JPY)       |
| 6       | New Zealand   | New Zealand dollar (NZD) |
| OT      | Singapore     | Singapore dollar (SGD)   |


### NASDAQ and NYSE Datasets
The [NASDAQ and NYSE Datasets](https://dl.acm.org/doi/abs/10.1145/3309547) offer daily stock features, including opening price, closing price, trading volume, lowest price, and highest price. The NASDAQ dataset has 1,244 time steps, while the NYSE dataset includes 1,243 time steps, each corresponding to one trading day.

### NN5 Dataset
The [NN5 Dataset](http://www.neural-forecasting-competition.com/NN5/) records daily cash demand at 111 ATMs across various locations in England from March 18, 1996, to March 22, 1998, along with an additional 8 weeks of test data. It consists of 791 time steps, recorded daily.

### ILI Dataset
The [ILI Dataset](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html) contains weekly records of influenza-like illness reported by the seven Centers for Disease Control and Prevention (CDC) across the United States from 2002 to 2021. It includes 966 time steps, each representing one week. 
| Variable          | Description                                            |
| ----------------- | ------------------------------------------------------ |
| % WEIGHTED ILI    | Percentage of ILI patients weighted by population size |
| % UNWEIGHTED ILI  | Unweighted percentage of ILI patients                  |
| AGE 0-4           | Number of 0-4 year old ILI patients                    |
| AGE 5-24          | Number of 5-24 year old ILI patients                   |
| ILITOTAL          | Total number of ILI patients                           |
| NUM. OF PROVIDERS | Number of healthcare providers                         |
| OT                | Total number of patients                               |

### Covid-19 Dataset
The [Covid-19 Dataset](https://ojs.aaai.org/index.php/AAAI/article/view/16616) documents daily new COVID-19 cases across 948 countries. It comprises 1,392 time steps recorded daily, spanning from January 3, 2020, to October 25, 2023.

### Wike2000 Dataset
The [Wike2000 Dataset](https://proceedings.mlr.press/v89/gasthaus19a.html) consists of daily page view counts for 2,000 Wikipedia pages from January 1, 2012, to March 2, 2014. It includes 792 time steps, each with a daily granularity.

---