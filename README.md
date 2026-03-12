# SaaS Atlas Platform (v3.0) 💠

The **SaaS Atlas Platform** is a professional-grade analytical tool designed for biological and physical data processing. It provides a robust, multi-tenant environment for automated curve fitting, high-dimensional visualization, and advanced statistical analysis.

---

## 🚀 Key Features

### 📉 **2D Trend Analysis**
- **Interactive Visualization:** Integrated **Plotly** support for real-time zooming, panning, and hovering.
- **Bi-directional Error Bars:** Support for both vertical (Y) and horizontal (X) error bars derived from independent SD/error sources.
- **Advanced Curve Fitting:** Automate non-linear fitting for **Sigmoid, Exponential, and Linear** models.
- **Machine Learning Fitting:** Use **SVR** (Support Vector Regression) or **Random Forest** for complex non-parametric trends.
- **Confidence Intervals (CI):** Optimized linear regression fitting that calculates and shades 95% Confidence Interval bands.

### 🧊 **3D Spatial Insights**
- Dual-engine rendering using both **Plotly (Interactive)** and **Matplotlib (Static)**.
- Multiple 3D plot types: **Scatter, Mesh, and Bar plots**.
- High-resolution data mapping with custom Z-axis selection.

### 📊 **Statistical Analysis Suite**
- **Pairwise Comparisons:** Automatic testing between groups (e.g., cell types or stage).
- **Available Tests:** 
  - **Welch’s t-test** (for unequal variance)
  - **Student’s t-test** (for equal variance)
  - **One-Way ANOVA**
- **P-Value Matrix Generation:** Automated generation of a comparison matrix and full statistical summary tables.

### 🔬 **Machine Learning & Data Management**
- **Unsupervised Clustering:** Automated **K-Means** clustering to identify data subgroups.
- **Multi-Sheet Excel Support:** Selective sheet importing from standard Excel workbooks.
- **Encoding-Aware CSV Loader:** Robust handling for scientific data containing special symbols (like "µ").

---

## 📥 Installation

### **Prerequisites**
- Python 3.10 or higher.
- A virtual environment is highly recommended.

### **Steps**
1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <project-folder>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the platform:**
   ```bash
   streamlit run SaaS_Atlas/app.py
   ```

---

## 📂 Project Structure
- `SaaS_Atlas/`: Core application logic and assets.
- `requirements.txt`: List of required Python packages.
- `output/`: (Auto-generated) Location for exported high-DPI plots and statistical reports.

---

## 🔬 Mathematical Basis

### **Sigmoid Fitting**
Used for modeling growth phases and saturation processes.
$f(t) = a + \frac{b}{1 + e^{-(t-t_0)/k}}$

### **Exponential Rise**
Used for kinetics that approach a steady-state value.
$f(x) = a(1 - e^{-x/b})$

### **Linear Regression**
Calculates constant rates of change and baseline drift.
$y = mx + c$

---

## 🎨 Visualization Controls
- **High-Resolution Export:** Save any plot at up to **1200 DPI**.
- **Custom Coloring:** Individually select colors for every dataset and feature trace via the sidebar color picker.
- **Exterior Legends:** All plot legends are positioned outside the plot box to maximize data visibility.

---
**Atlas Analytics Portal v3.0 | 2026 Enterprise Edition**
