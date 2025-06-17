# Property Price Prediction and Market Analysis System

## Overview
This is a comprehensive property price prediction and market analysis system built with Flask, Python, and modern web technologies. The system helps users analyze property markets, predict property prices, and visualize market trends.

## Features

### 1. Market Analysis Dashboard
- **Interactive Visualizations**: 
  - Price Distribution Chart
  - Price vs Area Scatter Plot
  - Average Price by Location
  - Average Price by Property Type
- **Key Statistics**:
  - Total Properties
  - Average Price
  - Average Area
  - Price per Square Foot
  - Median Price
  - Price Range
  - ![Screenshot 2025-05-04 105357](https://github.com/user-attachments/assets/6ed5e06f-76a7-4123-913c-e68fe6108373)


### 2. Interactive Map
- **Property Location Visualization**:
  - Cluster markers for better visualization
  - Property details on click
  - Color-coded by property type
- **Filtering Options**:
  - Price Range
  - Number of Bedrooms
  - Property Type
  - Location
  - ![Screenshot 2025-05-04 105233](https://github.com/user-attachments/assets/e67fbbbd-2bfb-4c43-8dae-70e14b09efc9)


### 3. Price Prediction
- **Smart Price Estimation**:
  - Input property details
  - Get instant price predictions
  - Factors considered:
    - Area
    - Location
    - Number of Bedrooms
    - Number of Bathrooms
    - Property Type
    - ![Screenshot 2025-05-04 105628](https://github.com/user-attachments/assets/8b58657a-c6ea-4682-93ff-c73ed045cef3)


## Technical Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Data Visualization**: Plotly.js
- **Mapping**: Folium
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy

## Installation

1. **Prerequisites**:
   - Python 3.8 or higher
   - pip (Python package manager)

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Application**:
   Open your web browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## Usage Guide

### Market Analysis
1. Navigate to the Market Analysis page
2. View interactive charts and statistics
3. Hover over charts for detailed information
4. Use filters to analyze specific market segments

### Interactive Map
1. Go to the Map page
2. View all properties on the map
3. Click on markers to see property details
4. Use filters to focus on specific properties

### Price Prediction
1. Access the Price Prediction feature
2. Enter property details:
   - Area (in square feet)
   - Location
   - Number of Bedrooms
   - Number of Bathrooms
   - Property Type
3. Get instant price prediction

## Data Sources
- Sample property data with 50 properties
- Randomly generated data with realistic ranges
- Properties located in Chennai, India
- Includes various property types and locations

## Features in Detail

### Market Analysis
- **Price Distribution**: Shows the distribution of property prices
- **Price vs Area**: Visualizes the relationship between price and area
- **Location Analysis**: Compares average prices across different locations
- **Property Type Analysis**: Analyzes price trends by property type

### Interactive Map Features
- **Cluster Markers**: Groups nearby properties for better visualization
- **Property Details**: Shows comprehensive information on click
- **Dynamic Filtering**: Real-time updates based on selected filters
- **Location-based Analysis**: Focus on specific areas or neighborhoods

### Price Prediction Model
- **Machine Learning**: Uses Random Forest algorithm
- **Multiple Features**: Considers various property attributes
- **Accurate Predictions**: Trained on realistic property data
- **Instant Results**: Quick price estimates

## Future Enhancements
1. User authentication and saved searches
2. Historical price trends
3. More detailed property information
4. Integration with real estate APIs
5. Mobile-responsive design improvements

## Troubleshooting
If you encounter any issues:
1. Check the console for error messages
2. Ensure all dependencies are installed
3. Verify the data file exists
4. Check if the required ports are available

## Support
For any questions or issues, please contact the development team. 
