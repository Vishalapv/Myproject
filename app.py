from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import folium
from folium.plugins import MarkerCluster
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from plotly.utils import PlotlyJSONEncoder
import json
import traceback

app = Flask(__name__)

# Add format_number filter
@app.template_filter('format_number')
def format_number(value):
    return "{:,}".format(int(value))

# Load and prepare data
def load_data():
    try:
        # Generate sample data with 50 properties
        np.random.seed(42)
        n_properties = 50
        
        # Generate random data
        data = {
            'Location': np.random.choice(['Adyar', 'Anna Nagar', 'Besant Nagar', 'Kilpauk', 'Mylapore', 'Nungambakkam', 'T Nagar'], n_properties),
            'Property_Type': np.random.choice(['Apartment', 'House'], n_properties),
            'Price': np.random.randint(10000000, 30000000, n_properties),
            'Area': np.random.randint(1000, 2000, n_properties),
            'No. of Bedrooms': np.random.randint(2, 5, n_properties),
            'No. of Bathrooms': np.random.randint(2, 4, n_properties),
            'Latitude': np.random.uniform(13.0, 13.1, n_properties),
            'Longitude': np.random.uniform(80.2, 80.3, n_properties)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Print the shape and columns for debugging
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['Price', 'Area', 'Latitude', 'Longitude', 'No. of Bedrooms', 'No. of Bathrooms']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure categorical columns are properly formatted
        categorical_columns = ['Location', 'Property_Type']
        for col in categorical_columns:
            df[col] = df[col].astype(str)
        
        # Handle missing values
        df = df.dropna(subset=['Latitude', 'Longitude', 'Price', 'Area'])
        
        print(f"Data loaded successfully. Shape after cleaning: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def prepare_data(df):
    # Separate features
    numerical_features = ['Area', 'No. of Bedrooms', 'No. of Bathrooms']
    categorical_features = ['Location', 'Property_Type']
    
    # Handle missing values
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    df[numerical_features] = numerical_imputer.fit_transform(df[numerical_features])
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    
    # Encode categorical features
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        df[feature] = label_encoders[feature].fit_transform(df[feature])
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df, label_encoders, scaler, numerical_features, categorical_features

def train_model(df):
    # Separate features and target
    X = df.drop(['Price'], axis=1)
    y = df['Price']
    
    # Define numerical and categorical features
    numerical_features = ['Area', 'No. of Bedrooms', 'No. of Bathrooms']
    categorical_features = ['Location', 'Property_Type']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create and train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    return model

# Load or train model
if os.path.exists('model.joblib'):
    model = joblib.load('model.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    scaler = joblib.load('scaler.joblib')
    numerical_features, categorical_features = joblib.load('features.joblib')
else:
    model = train_model(load_data())
    label_encoders = {}
    scaler = {}
    numerical_features = []
    categorical_features = []

# Create map
def create_map(df, filtered_df=None):
    try:
        if filtered_df is None:
            filtered_df = df
        
        print("Creating map with data shape:", filtered_df.shape)
        print("Columns in filtered data:", filtered_df.columns.tolist())
        
        # Ensure required columns exist
        required_columns = ['Latitude', 'Longitude', 'Location', 'Price', 'Area', 'No. of Bedrooms', 'No. of Bathrooms', 'Property_Type']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for any missing values in required columns
        missing_values = filtered_df[required_columns].isnull().sum()
        if missing_values.any():
            print("Warning: Missing values found:", missing_values[missing_values > 0])
            filtered_df = filtered_df.dropna(subset=required_columns)
            print("Data shape after removing missing values:", filtered_df.shape)
        
        # Create base map centered on Chennai
        m = folium.Map(location=[13.0827, 80.2707], zoom_start=12)
        
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each property
        for idx, row in filtered_df.iterrows():
            try:
                # Format the price with commas
                formatted_price = f"₹{row['Price']:,.0f}"
                
                popup_content = f"""
                <div style="font-family: Arial, sans-serif; font-size: 14px;">
                    <b>Location:</b> {row['Location']}<br>
                    <b>Price:</b> {formatted_price}<br>
                    <b>Area:</b> {row['Area']:,.0f} sq ft<br>
                    <b>Bedrooms:</b> {row['No. of Bedrooms']}<br>
                    <b>Bathrooms:</b> {row['No. of Bathrooms']}<br>
                    <b>Type:</b> {row['Property_Type']}
                </div>
                """
                
                # Create marker with popup
                folium.Marker(
                    location=[float(row['Latitude']), float(row['Longitude'])],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='blue' if row['Property_Type'] == 'Apartment' else 'green')
                ).add_to(marker_cluster)
            except Exception as e:
                print(f"Error creating marker for row {idx}: {str(e)}")
                continue
        
        print("Map created successfully")
        return m._repr_html_()
    except Exception as e:
        print(f"Error creating map: {str(e)}")
        error_html = f"""
        <div class="alert alert-danger">
            <h4>Error Creating Map</h4>
            <p>{str(e)}</p>
            <p>Please check the console for more details.</p>
        </div>
        """
        return error_html

# Create price distribution plot
def create_price_distribution_plot(df):
    # Create bins for the histogram
    min_price = df['Price'].min()
    max_price = df['Price'].max()
    bin_size = (max_price - min_price) / 30
    bins = [min_price + i * bin_size for i in range(31)]
    
    # Calculate counts for each bin
    counts = []
    for i in range(len(bins)-1):
        count = len(df[(df['Price'] >= bins[i]) & (df['Price'] < bins[i+1])])
        counts.append(count)
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)],
        y=counts,
        marker_color='#0d6efd',
        hovertemplate='Price: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Property Price Distribution',
        xaxis_title='Price (₹)',
        yaxis_title='Count',
        showlegend=False,
        template='plotly_white',
        margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
        height=400
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)

# Create price vs area plot
def create_price_vs_area_plot(df):
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Area'].tolist(),
        y=df['Price'].tolist(),
        mode='markers',
        marker=dict(
            color=df['Price'].tolist(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Price (₹)')
        ),
        text=df.apply(lambda row: f"Location: {row['Location']}<br>Type: {row['Property_Type']}", axis=1).tolist(),
        hovertemplate='Area: %{x} sq ft<br>Price: ₹%{y:,.0f}<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Price vs Area',
        xaxis_title='Area (sq ft)',
        yaxis_title='Price (₹)',
        template='plotly_white',
        margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
        height=400
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False)

# Create location price plot
def create_location_price_plot(df):
    try:
        print("\nCreating location price plot...")
        print("Available columns:", df.columns.tolist())
        
        # Calculate statistics by location
        location_stats = df.groupby('Location').agg({
            'Price': ['mean', 'count']
        }).reset_index()
        
        # Flatten the multi-index columns
        location_stats.columns = ['Location', 'Average Price', 'Count']
        
        print("Location stats calculated:", location_stats.to_dict())
        
        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=location_stats['Location'].tolist(),
            y=location_stats['Average Price'].tolist(),
            marker_color='#0d6efd',
            text=location_stats['Count'].tolist(),
            hovertemplate='Location: %{x}<br>Average Price: ₹%{y:,.0f}<br>Properties: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Average Price by Location',
            xaxis_title='Location',
            yaxis_title='Average Price (₹)',
            showlegend=False,
            template='plotly_white',
            margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
            height=400
        )
        
        print("Location price plot created successfully")
        return fig.to_html(full_html=False, include_plotlyjs=False)
        
    except Exception as e:
        print(f"Error in create_location_price_plot: {str(e)}")
        print("DataFrame info:")
        print(df.info())
        raise

# Create property type plot
def create_property_type_plot(df):
    try:
        print("\nCreating property type plot...")
        print("Available columns:", df.columns.tolist())
        
        # Calculate statistics by property type
        type_stats = df.groupby('Property_Type').agg({
            'Price': ['mean', 'count'],
            'Area': 'mean'
        }).reset_index()
        
        # Flatten the multi-index columns
        type_stats.columns = ['Property Type', 'Average Price', 'Count', 'Average Area']
        
        print("Type stats calculated:", type_stats.to_dict())
        
        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=type_stats['Property Type'].tolist(),
            y=type_stats['Average Price'].tolist(),
            marker_color='#0d6efd',
            text=type_stats.apply(lambda row: f"Count: {row['Count']}<br>Avg Area: {row['Average Area']:,.0f} sq ft", axis=1).tolist(),
            hovertemplate='Type: %{x}<br>Average Price: ₹%{y:,.0f}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Average Price by Property Type',
            xaxis_title='Property Type',
            yaxis_title='Average Price (₹)',
            showlegend=False,
            template='plotly_white',
            margin={'l': 50, 'r': 50, 't': 50, 'b': 50},
            height=400
        )
        
        print("Property type plot created successfully")
        return fig.to_html(full_html=False, include_plotlyjs=False)
        
    except Exception as e:
        print(f"Error in create_property_type_plot: {str(e)}")
        print("DataFrame info:")
        print(df.info())
        raise

@app.route('/')
def home():
    try:
        print("Loading home page...")
        print("Current working directory:", os.getcwd())
        print("Templates directory:", os.path.join(os.getcwd(), 'templates'))
        print("Template files:", os.listdir('templates'))
        return render_template('index.html')
    except Exception as e:
        print(f"Error loading home page: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return str(e)

@app.route('/map')
def map_page():
    try:
        print("Loading data for map page...")
        # Load the data
        df = load_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Get unique locations for the filter
        locations = sorted(df['Location'].unique().tolist())
        print(f"Found {len(locations)} unique locations")
        
        # Create the map
        print("Creating map...")
        map_html = create_map(df)
        
        # Calculate statistics
        total_properties = len(df)
        avg_price = df['Price'].mean()
        avg_area = df['Area'].mean()
        print(f"Statistics calculated - Total properties: {total_properties}, Avg price: {avg_price}, Avg area: {avg_area}")
        
        return render_template('map.html',
                             map_html=map_html,
                             locations=locations,
                             total_properties=total_properties,
                             avg_price=avg_price,
                             avg_area=avg_area)
    except Exception as e:
        print(f"Error in map_page: {str(e)}")
        error_message = f"""
        <div class="container mt-4">
            <div class="alert alert-danger">
                <h4>Error Loading Map</h4>
                <p>{str(e)}</p>
                <p>Please check the console for more details.</p>
            </div>
        </div>
        """
        return error_message

@app.route('/filter_properties', methods=['POST'])
def filter_properties():
    try:
        print("Processing filter request...")
        df = load_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Get filter parameters from JSON data
        data = request.get_json()
        print("Received filter parameters:", data)
        
        price_range = data.get('priceRange', 'all')
        bedrooms = data.get('bedrooms', 'all')
        property_type = data.get('propertyType', 'all')
        location = data.get('location', 'all')
        
        # Apply filters
        filtered_df = df.copy()
        
        if price_range != 'all':
            min_price, max_price = map(int, price_range.split('-'))
            filtered_df = filtered_df[(filtered_df['Price'] >= min_price) & (filtered_df['Price'] <= max_price)]
            print(f"Applied price range filter: {min_price} - {max_price}, Remaining properties: {len(filtered_df)}")
        
        if bedrooms != 'all':
            if bedrooms == '4+':
                filtered_df = filtered_df[filtered_df['No. of Bedrooms'] >= 4]
            else:
                filtered_df = filtered_df[filtered_df['No. of Bedrooms'] == int(bedrooms)]
            print(f"Applied bedrooms filter: {bedrooms}, Remaining properties: {len(filtered_df)}")
        
        if property_type != 'all':
            filtered_df = filtered_df[filtered_df['Property_Type'] == property_type]
            print(f"Applied property type filter: {property_type}, Remaining properties: {len(filtered_df)}")
        
        if location != 'all':
            filtered_df = filtered_df[filtered_df['Location'] == location]
            print(f"Applied location filter: {location}, Remaining properties: {len(filtered_df)}")
        
        # Create filtered map
        print("Creating filtered map...")
        map_html = create_map(df, filtered_df)
        
        # Calculate statistics
        stats = {
            'total_properties': len(filtered_df),
            'average_price': int(filtered_df['Price'].mean()) if not filtered_df.empty else 0,
            'average_area': int(filtered_df['Area'].mean()) if not filtered_df.empty else 0,
            'price_per_sqft': int(filtered_df['Price'].mean() / filtered_df['Area'].mean()) if not filtered_df.empty else 0
        }
        print("Calculated statistics:", stats)
        
        return jsonify({
            'success': True,
            'map_html': map_html,
            'stats': stats
        })
    except Exception as e:
        print(f"Error in filter_properties: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/price_distribution')
def price_distribution():
    try:
        print("Loading data for price distribution...")
        df = load_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Calculate statistics
        stats = {
            'total_properties': len(df),
            'average_price': int(df['Price'].mean()),
            'average_area': int(df['Area'].mean()),
            'price_per_sqft': int(df['Price'].mean() / df['Area'].mean()),
            'median_price': int(df['Price'].median()),
            'min_price': int(df['Price'].min()),
            'max_price': int(df['Price'].max()),
            'average_bedrooms': float(df['No. of Bedrooms'].mean()),
            'average_bathrooms': float(df['No. of Bathrooms'].mean())
        }
        print("Statistics calculated:", stats)
        
        # Create price distribution plot
        print("Creating price distribution plot...")
        price_dist_plot = create_price_distribution_plot(df)
        
        # Create price vs area plot
        print("Creating price vs area plot...")
        price_vs_area_plot = create_price_vs_area_plot(df)
        
        # Create location price plot
        print("Creating location price plot...")
        location_price_plot = create_location_price_plot(df)
        
        # Create property type plot
        print("Creating property type plot...")
        property_type_plot = create_property_type_plot(df)
        
        print("All plots created successfully")
        
        return render_template('price_distribution.html',
                             stats=stats,
                             price_dist_plot=price_dist_plot,
                             price_vs_area_plot=price_vs_area_plot,
                             location_price_plot=location_price_plot,
                             property_type_plot=property_type_plot)
    except Exception as e:
        print(f"Error in price_distribution: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return render_template('error.html', error=str(e))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert string values to appropriate types
        property_data = {
            'Area': int(data.get('area', 0)),
            'No. of Bedrooms': int(data.get('bedrooms', 0)),
            'No. of Bathrooms': int(data.get('bathrooms', 0)),
            'Location': data.get('location', ''),
            'Property_Type': 'Apartment'  # Default to Apartment
        }
        
        # Create DataFrame with the same structure as training data
        df = pd.DataFrame([property_data])
        
        # Transform input data
        if hasattr(model, 'named_steps'):
            # If using a pipeline
            prediction = model.predict(df)[0]
        else:
            # If using separate scaler and encoders
            df[numerical_features] = scaler.transform(df[numerical_features])
            for feature in categorical_features:
                if feature in label_encoders:
                    df[feature] = label_encoders[feature].transform(df[feature])
            prediction = model.predict(df[numerical_features + categorical_features])[0]
        
        # Format the prediction as currency
        formatted_prediction = "₹{:,.0f}".format(prediction)
        
        return jsonify({
            'success': True,
            'prediction': formatted_prediction
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Add this line for debugging
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/estimate_price', methods=['POST'])
def estimate_price():
    try:
        data = request.json
        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        location = data['location']
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Area': [area],
            'No. of Bedrooms': [bedrooms],
            'No. of Bathrooms': [bathrooms],
            'Location': [location],
            'Property_Type': ['Apartment']  # Default to Apartment
        })
        
        # Transform input data
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        for feature in categorical_features:
            input_data[feature] = label_encoders[feature].transform(input_data[feature])
        
        # Make prediction
        prediction = model.predict(input_data[numerical_features + categorical_features])[0]
        
        return jsonify({
            'success': True,
            'estimated_price': int(prediction)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/locations')
def get_locations():
    df = load_data()
    locations = sorted(df['Location'].unique().tolist())
    return jsonify({
        'success': True,
        'locations': locations
    })

@app.route('/price_distribution_data')
def price_distribution_data():
    df = load_data()
    
    # Create price distribution plot
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Price'],
        nbinsx=30,
        name='Price Distribution',
        marker_color='#0d6efd'
    ))
    
    fig.update_layout(
        title='Property Price Distribution',
        xaxis_title='Price (₹)',
        yaxis_title='Count',
        showlegend=False,
        template='plotly_white'
    )
    
    return jsonify({
        'success': True,
        'plot': fig.to_json()
    })

@app.route('/price_trends')
def price_trends():
    df = load_data()
    
    # Generate sample date range
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    locations = df['Location'].unique()
    
    # Create price trends plot
    fig = go.Figure()
    
    for location in locations:
        location_data = df[df['Location'] == location]
        avg_price = location_data['Price'].mean()
        
        # Generate trend data with some random variation
        base_prices = [avg_price * (1 + np.random.normal(0, 0.05)) for _ in range(len(dates))]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=base_prices,
            mode='lines+markers',
            name=location,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Price Trends by Location',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    return jsonify({
        'success': True,
        'plot': fig.to_json()
    })

@app.route('/neighborhood_comparison')
def neighborhood_comparison():
    df = load_data()
    
    # Calculate average price by location
    location_stats = df.groupby('Location').agg({
        'Price': ['mean', 'median', 'min', 'max']
    }).reset_index()
    
    location_stats.columns = ['Location', 'Average Price', 'Median Price', 'Min Price', 'Max Price']
    
    # Create neighborhood comparison plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=location_stats['Location'],
        y=location_stats['Average Price'],
        name='Average Price',
        marker_color='#0d6efd'
    ))
    
    fig.update_layout(
        title='Average Property Prices by Neighborhood',
        xaxis_title='Neighborhood',
        yaxis_title='Price (₹)',
        template='plotly_white',
        showlegend=False
    )
    
    return jsonify({
        'success': True,
        'plot': fig.to_json()
    })

@app.route('/property_type_analysis')
def property_type_analysis():
    df = load_data()
    
    # Calculate statistics by property type
    property_stats = df.groupby('Property_Type').agg({
        'Price': ['mean', 'count'],
        'Area': 'mean'
    }).reset_index()
    
    property_stats.columns = ['Property Type', 'Average Price', 'Count', 'Average Area']
    property_stats['Price per sq ft'] = property_stats['Average Price'] / property_stats['Average Area']
    property_stats['Market Share'] = (property_stats['Count'] / property_stats['Count'].sum() * 100).round(1)
    
    # Create property type analysis plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=property_stats['Property Type'],
        y=property_stats['Average Price'],
        name='Average Price',
        marker_color='#0d6efd'
    ))
    
    fig.update_layout(
        title='Average Price by Property Type',
        xaxis_title='Property Type',
        yaxis_title='Price (₹)',
        template='plotly_white',
        showlegend=False
    )
    
    return jsonify({
        'success': True,
        'plot': fig.to_json(),
        'stats': property_stats.to_dict('records')
    })

@app.route('/market_stats')
def market_stats():
    try:
        df = load_data()
        
        stats = {
            'total_properties': len(df),
            'average_price': int(df['Price'].mean()),
            'median_price': int(df['Price'].median()),
            'min_price': int(df['Price'].min()),
            'max_price': int(df['Price'].max())
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/market_analysis')
def market_analysis():
    try:
        print("Starting market analysis...")
        df = load_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        
        # Calculate statistics
        stats = {
            'total_properties': len(df),
            'average_price': int(df['Price'].mean()),
            'average_area': int(df['Area'].mean()),
            'price_per_sqft': int(df['Price'].mean() / df['Area'].mean()),
            'median_price': int(df['Price'].median()),
            'min_price': int(df['Price'].min()),
            'max_price': int(df['Price'].max()),
            'average_bedrooms': float(df['No. of Bedrooms'].mean()),
            'average_bathrooms': float(df['No. of Bathrooms'].mean())
        }
        print("Statistics calculated:", stats)
        
        # Create plots
        print("Creating price distribution plot...")
        price_dist_plot = create_price_distribution_plot(df)
        
        print("Creating price vs area plot...")
        price_vs_area_plot = create_price_vs_area_plot(df)
        
        print("Creating location price plot...")
        location_price_plot = create_location_price_plot(df)
        
        print("Creating property type plot...")
        property_type_plot = create_property_type_plot(df)
        
        print("All plots created successfully")
        
        return render_template('market_analysis.html',
                             stats=stats,
                             price_dist_plot=price_dist_plot,
                             price_vs_area_plot=price_vs_area_plot,
                             location_price_plot=location_price_plot,
                             property_type_plot=property_type_plot)
    except Exception as e:
        print(f"Error in market_analysis: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return render_template('error.html', error=str(e))

@app.route('/property_stats')
def property_stats():
    try:
        print("Loading property statistics...")
        df = pd.read_csv('Book_PREDICT_with_coords.csv')
        
        # Basic statistics
        stats = {
            'total_properties': len(df),
            'average_price': int(df['SALES PRICE'].mean()),
            'average_area': int(df['INT_SQFT'].mean()),
            'price_per_sqft': int(df['SALES PRICE'].mean() / df['INT_SQFT'].mean()),
            'min_price': int(df['SALES PRICE'].min()),
            'max_price': int(df['SALES PRICE'].max())
        }
        
        # Create a simple price distribution plot
        price_bins = np.linspace(df['SALES PRICE'].min(), df['SALES PRICE'].max(), 20)
        price_counts, _ = np.histogram(df['SALES PRICE'], bins=price_bins)
        
        plot_data = {
            'data': [{
                'type': 'bar',
                'x': price_bins[:-1].tolist(),
                'y': price_counts.tolist(),
                'marker': {'color': '#0d6efd'},
                'hovertemplate': 'Price: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>'
            }],
            'layout': {
                'title': 'Property Price Distribution',
                'xaxis': {'title': 'Price (₹)'},
                'yaxis': {'title': 'Number of Properties'},
                'showlegend': False,
                'template': 'plotly_white'
            }
        }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'plot': plot_data
        })
    except Exception as e:
        print(f"Error in property_stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Current working directory:", os.getcwd())
    print("Template directory exists:", os.path.exists('templates'))
    if os.path.exists('templates'):
        print("Template files:", os.listdir('templates'))
    app.run(debug=True, host='0.0.0.0', port=5000) 