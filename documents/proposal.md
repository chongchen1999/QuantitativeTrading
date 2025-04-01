## **Final Project Proposal: Transformer-Based Models for Enhancing Quantitative Trading Strategies**  

#### **Team Members**: Chong Chen, Di Liu

---

### **1. Subject**  

In this project, we will explore the application of transformer-based models in quantitative trading. Traditional financial models, such as moving averages and factor-based approaches, have limitations in capturing complex, time series information of the market. Meanwhile, deep learning techniques like LSTMs and GRUs have shown promise but struggle with long-term dependencies and scalability.

Our goal is to develop a **transformer-based architecture** tailored for financial time series prediction, leveraging **self-attention mechanisms** to extract meaningful patterns from historical stock data. The model will be trained to predict future stock returns based on historical price movements and trading volume data. To validate its effectiveness, we will test it on the **U.S. stock market** (e.g., S&P 500, NASDAQ) and compare its performance with traditional models.

---

### **2. Data Collection**  

#### **Datasets**  
- **Historical Stock Data**  
  - **Scope**: Daily, weekly, and monthly adjusted closing prices, turnover rates, and trading volumes for U.S. stocks (e.g., S&P 500, NASDAQ).  
  - **Sources**: Yahoo Finance, Alpha Vantage, or Kaggle.  
- **Benchmark Data**  
  - S&P 500 index data to compute excess returns and risk-adjusted metrics.  

#### **Preprocessing**  
- Handle missing values, including delisted or suspended stocks.  
- Normalize features using zero-mean, unit-variance normalization.  
- Segment data into training (e.g., 2010–2020) and testing (e.g., 2021–2024) periods for robust evaluation.  

---

### **3. Methods**  

#### **Model Architecture**  
To leverage the advantages of transformers in sequence modeling, we propose a **custom transformer-based architecture** for financial time series forecasting:
- Apply **multi-head self-attention** to capture relationships across different stocks and time horizons.  
- Modify the **decoder** for direct regression and classification of stock returns.  
- Train using **MSE loss** and optimize with **Adam** to enhance convergence.  

#### **Baselines for Comparison**  
- **Traditional Machine Learning Models**: LSTM, GRU.  
- **Classical Quantitative Strategies**: Momentum, moving averages, mean reversion models.  

#### **Experiments**  
- Test performance across different data frequencies (daily, weekly, monthly).  
- Analyze model scalability by varying stock selection criteria (e.g., top 1%, 5%, 10% of stocks by market cap).  


#### **Backtesting Framework**  
To assess real-world applicability, we will simulate trading strategies using model-generated signals. Evaluation metrics include:  
- **Sharpe Ratio** (risk-adjusted return measure).  
- **Alpha** (excess return relative to the market benchmark).  
- **Value at Risk (VaR)** (downside risk assessment).  
- **Maximum Drawdown** (worst-case loss scenario).  

---

### **4. Goals**  

#### **Primary Objectives**  
- Develop and validate a **transformer-based model** for predicting stock returns.  
- Demonstrate superior risk-adjusted returns (**Sharpe Ratio > 1.0, positive Alpha**) compared to traditional approaches.  

#### **Secondary Objectives**  
- Investigate the impact of different training frequencies (daily vs. monthly) and dataset scales on predictive performance.  
- Open-source our model, code, and results for transparency and reproducibility.  

#### **Stretch Goal**  
- Enhance the model by integrating alternative data sources (e.g., news sentiment, macroeconomic indicators) to improve predictive accuracy.  

---

### **5. Expected Outcome**  
- A novel transformer-based trading strategy that effectively captures complex market patterns.  
- A comparative analysis of our model against traditional financial models and deep learning baselines.  
