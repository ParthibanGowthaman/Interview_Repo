"""
Centralized Warehouse Asking Rent Forecasting Script - Quarter Enhancement Challenge
===================================================================================
CANDIDATE TASK: 
The script currently only supports yearly forecast start/end dates. Your task is to:
1. Add support for specifying quarters (Q1-Q4) in addition to years
2. Implement FORECAST_START_QUARTER and FORECAST_END_QUARTER configuration
3. Ensure the forecast data filters correctly based on both year AND quarter
4. Update all relevant date filtering logic throughout the script

Example: 
- Current: FORECAST_START_YEAR = 2025, FORECAST_END_YEAR = 2030
- Required: Add FORECAST_START_QUARTER = 2, FORECAST_END_QUARTER = 3
- Result: Forecast from 2025 Q2 to 2030 Q3

BUGS TO FIX:
- Line 59-60: Missing quarter configuration variables
- Line 476-479: Date filtering doesn't consider quarters
- Line 546, 619-662: Special handling for first forecast period needs quarter logic
- Various other places where quarter filtering should be applied

Author: Interview Candidate
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
import glob
from datetime import datetime, timedelta
import json
import traceback
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from statsmodels.tsa.filters.hp_filter import hpfilter

warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
log_filename = f'centralized_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_ASKING_RENT = 30.0
DATE_TOLERANCE_DAYS = 45
MIN_FORECAST_PERIODS = 4

# BUG #1: Missing quarter configuration - candidates need to add these
FORECAST_START_YEAR = 2025
FORECAST_END_YEAR = 2030
# TODO: Add FORECAST_START_QUARTER and FORECAST_END_QUARTER variables here
# FORECAST_START_QUARTER = ???  # Should be 1, 2, 3, or 4
# FORECAST_END_QUARTER = ???    # Should be 1, 2, 3, or 4

# INPUT FILE CONFIGURATION
if 'FORECAST_SCENARIO_FILE' in os.environ:
    FUTURE_ECONOMIC_DATA_FILE = os.environ['FORECAST_SCENARIO_FILE']
else:
    FUTURE_ECONOMIC_DATA_FILE = 'FutureForecastData_complete_Base.csv'

HISTORICAL_DATA_FILE = 'RawComplete_GDP_Filled.csv'

# Extract scenario name dynamically from filename
def extract_scenario_name(filename):
    """Extract scenario name from filename pattern"""
    base_name = os.path.basename(filename).replace('.csv', '')
    
    if 'complete_' in base_name:
        parts = base_name.split('complete_')
        if len(parts) > 1:
            scenario = parts[1].replace('_', ' ').title()
            return scenario
    
    if 'base' in base_name.lower():
        return 'Base'
    elif 'mod' in base_name.lower() and 'down' in base_name.lower():
        return 'Moderate Downside'
    elif 'sev' in base_name.lower() and 'down' in base_name.lower():
        return 'Severe Downside'
    elif 'upside' in base_name.lower():
        return 'Upside'
    
    return base_name.replace('FutureForecastData_', '').replace('_', ' ').title()

SCENARIO_NAME = extract_scenario_name(FUTURE_ECONOMIC_DATA_FILE)

# CAPPING CONFIGURATION
USE_SD_CAPPING = True
if 'FORECAST_SD_MULTIPLIER' in os.environ:
    SD_MULTIPLIER = float(os.environ['FORECAST_SD_MULTIPLIER'])
else:
    SD_MULTIPLIER = 1.0

MIN_HISTORICAL_PERIODS = 20

# BUG #2: This print statement needs to include quarter information
print(f"🌐 CENTRALIZED MULTI-METRO ASKING RENT FORECASTING ({FORECAST_START_YEAR}-{FORECAST_END_YEAR})")
# TODO: Update to show quarters, e.g., "2025 Q2 - 2030 Q3"
print("=" * 70)
logger.info(f"Log file created: {log_filename}")

def quarter_to_month(quarter):
    """
    Convert quarter number to month number
    Q1 = 1 (January), Q2 = 4 (April), Q3 = 7 (July), Q4 = 10 (October)
    """
    # BUG #3: This function is incomplete
    # TODO: Implement proper quarter to month conversion
    pass

def is_date_in_forecast_range(date, year, month):
    """
    Check if a date falls within the forecast range including quarters
    
    Args:
        date: The date to check
        year: Year from the data
        month: Month from the data (1, 4, 7, or 10 for quarters)
    
    Returns:
        bool: True if date is within forecast range
    """
    # BUG #4: This function needs to be implemented
    # TODO: Check if the date is between start year/quarter and end year/quarter
    # Remember: months 1,4,7,10 correspond to Q1,Q2,Q3,Q4
    pass

def find_latest_results_folder() -> Optional[str]:
    """Find the most recent multi_metro_hierarchical_results folder"""
    folders = glob.glob('multi_metro_hierarchical_results_*')
    if not folders:
        logger.error("No multi-metro results folders found")
        return None
    
    folders.sort(reverse=True)
    latest = folders[0]
    logger.info(f"Found latest results folder: {latest}")
    return latest

def validate_model_package(model_package: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that model package has all required components"""
    required_keys = ['model_info', 'feature_engineering', 'historical_context', 'metadata']
    
    for key in required_keys:
        if key not in model_package:
            return False, f"Missing required key: {key}"
    
    model_info = model_package['model_info']
    required_model_keys = ['name', 'type', 'features', 'intercept', 'coefficients', 'performance']
    for key in required_model_keys:
        if key not in model_info:
            return False, f"Missing model_info key: {key}"
    
    features = model_info['features']
    coefficients = model_info['coefficients']
    if len(features) != len(coefficients):
        return False, f"Feature count ({len(features)}) doesn't match coefficient count ({len(coefficients)})"
    
    return True, "Valid"

def load_metro_model(metro_folder: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Load the model package for a specific metro with validation"""
    model_path = os.path.join(metro_folder, 'warehouse_asking_rent_model_package.pkl')
    
    if not os.path.exists(model_path):
        logger.warning(f"Model package not found at {model_path}")
        return None, f"Model package not found"
    
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        is_valid, message = validate_model_package(model_package)
        if not is_valid:
            logger.error(f"Invalid model package: {message}")
            return None, f"Invalid model package: {message}"
            
        return model_package, None
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None, f"Error loading model: {str(e)}"

def create_features_for_metro(future_data: pd.DataFrame, 
                            model_info: Dict[str, Any], 
                            feature_config: Dict[str, Any]) -> pd.DataFrame:
    """Create features for a specific metro based on model requirements"""
    
    features_needed = model_info['features']
    logger.debug(f"Creating features: {features_needed}")
    
    feature_to_column_map = {
        'Bond_Yield': 'Bond_Yield_Abs_Change_YoY',
        'CPI': 'CPI_Pct_Change_YoY',
        'Employment_Transport': 'Employment_Transport_Pct_Change_YoY',
        'GDP_Transport': 'GDP_Transport_Real_Pct_Change_YoY',
        'Retail_Sales': 'Retail_Sales_Real_Pct_Change_YoY',
        'Consumer_Spending': 'Consumer_Spending_Real_Pct_Change_YoY',
        'Vacancy_Ratio': 'Vacancy_Ratio'
    }
    
    data = future_data.copy()
    
    required_base_cols = ['10-Yr Bond Yield', 'Consumer price index', 
                         'Employment - Utilities, transportation and warehousing',
                         'GDP, real - Transportation and warehousing',
                         'Retail sales (excluding sales of vehicles), real',
                         'Consumer spending, real', 'Vac %']
    missing_cols = [col for col in required_base_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    numeric_cols = ['10-Yr Bond Yield', 'Consumer price index', 
                   'Employment - Utilities, transportation and warehousing',
                   'GDP, real - Transportation and warehousing',
                   'Retail sales (excluding sales of vehicles), real',
                   'Consumer spending, real']
    for col in numeric_cols:
        if col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str).str.replace(',', '')
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    if 'Bond_Yield_Abs_Change_YoY' not in data.columns:
        logger.debug("Creating YoY features")
        data['Bond_Yield_Abs_Change_YoY'] = data['10-Yr Bond Yield'].diff(4)
        data['CPI_Pct_Change_YoY'] = data['Consumer price index'].pct_change(4) * 100
        data['Employment_Transport_Pct_Change_YoY'] = data['Employment - Utilities, transportation and warehousing'].pct_change(4) * 100
        data['GDP_Transport_Real_Pct_Change_YoY'] = data['GDP, real - Transportation and warehousing'].pct_change(4) * 100
        data['Retail_Sales_Real_Pct_Change_YoY'] = data['Retail sales (excluding sales of vehicles), real'].pct_change(4) * 100
        data['Consumer_Spending_Real_Pct_Change_YoY'] = data['Consumer spending, real'].pct_change(4) * 100
        
        data['GDP_Transport_Pct_Change_YoY'] = data['GDP_Transport_Real_Pct_Change_YoY']
        data['Retail_Sales_Pct_Change_YoY'] = data['Retail_Sales_Real_Pct_Change_YoY']
        data['Consumer_Spending_Pct_Change_YoY'] = data['Consumer_Spending_Real_Pct_Change_YoY']
    
    if 'Vacancy_Ratio' in data.columns and not data['Vacancy_Ratio'].isna().all():
        logger.debug(f"Using pre-calculated Vacancy Ratio. Range: {data['Vacancy_Ratio'].min():.3f} to {data['Vacancy_Ratio'].max():.3f}")
        data['Vacancy_Ratio'] = pd.to_numeric(data['Vacancy_Ratio'], errors='coerce')
    else:
        logger.debug("Vacancy_Ratio not found or empty in data. Calculating using HP filter")
        data['Vac_Numeric'] = pd.to_numeric(data['Vac %'].str.replace('%', ''), errors='coerce')
        
        vacancy_clean = data['Vac_Numeric'].dropna()
        if len(vacancy_clean) > 10:
            try:
                from statsmodels.tsa.filters.hp_filter import hpfilter
                cycle, natural_vacancy = hpfilter(vacancy_clean.values, lamb=20000)
                data['Natural_Vacancy'] = np.nan
                data.loc[vacancy_clean.index, 'Natural_Vacancy'] = natural_vacancy
                data['Vacancy_Ratio'] = data['Vac_Numeric'] / data['Natural_Vacancy']
                logger.debug(f"HP filter successful. Calculated Vacancy Ratio range: {data['Vacancy_Ratio'].min():.3f} to {data['Vacancy_Ratio'].max():.3f}")
            except Exception as e:
                logger.warning(f"HP filter failed: {e}. Using simple moving average.")
                data['Natural_Vacancy'] = data['Vac_Numeric'].rolling(window=12, center=True).mean()
                data['Vacancy_Ratio'] = data['Vac_Numeric'] / data['Natural_Vacancy']
        else:
            logger.warning("Insufficient data for HP filter, using mean vacancy")
            data['Natural_Vacancy'] = data['Vac_Numeric'].mean()
            data['Vacancy_Ratio'] = data['Vac_Numeric'] / data['Natural_Vacancy']
    
    # Create lag features
    for lag in [1, 2, 3]:
        for base_feat, source_col in feature_to_column_map.items():
            lag_feat = f'{base_feat}_lag{lag}'
            if lag_feat not in data.columns and source_col in data.columns:
                data[lag_feat] = data[source_col].shift(lag)
                logger.debug(f"Created lag feature: {lag_feat}")
    
    # Create moving average features
    for base_feat, source_col in feature_to_column_map.items():
        ma_feat = f'{base_feat}_ma3'
        if ma_feat not in data.columns and source_col in data.columns:
            temp_ma = data[source_col].rolling(window=3, min_periods=3).mean()
            data[ma_feat] = temp_ma.shift(1)
            logger.debug(f"Created MA feature: {ma_feat}")
    
    # Additional feature engineering code continues...
    # [Truncated for brevity - includes momentum, trend, regime indicators, interactions]
    
    missing_features = [f for f in features_needed if f not in data.columns]
    if missing_features:
        logger.warning(f"Missing features after creation: {missing_features}")
    
    return data

def calculate_metro_bounds(metro_code: str, 
                         historical_data: pd.DataFrame) -> Tuple[float, float, float, float, bool]:
    """
    Calculate metro-specific bounds for YoY changes using historical standard deviation
    """
    if pd.isna(metro_code) or metro_code == 'NA':
        metro_code = 'NA'
    
    metro_historical = historical_data[historical_data['Metro Code'] == metro_code].copy()
    
    if len(metro_historical) == 0:
        safe_metro_code = metro_code.replace('NA_Metro', 'NA') if isinstance(metro_code, str) else 'NA'
        metro_historical = historical_data[historical_data['Metro Code'] == safe_metro_code].copy()
    
    metro_historical = metro_historical.sort_values('date_col')
    metro_historical['AskingRent_YoY_Pct_Change'] = metro_historical['Asking Rent/SF'].pct_change(4) * 100
    
    baseline_data = metro_historical[(metro_historical['year'] >= 2002) & (metro_historical['year'] <= 2018)]
    valid_yoy_changes = baseline_data['AskingRent_YoY_Pct_Change'].dropna()
    
    if len(valid_yoy_changes) >= MIN_HISTORICAL_PERIODS:
        historical_mean_change = valid_yoy_changes.mean()
        historical_sd_change = valid_yoy_changes.std()
        
        lower_bound = historical_mean_change - (SD_MULTIPLIER * historical_sd_change)
        upper_bound = historical_mean_change + (SD_MULTIPLIER * historical_sd_change)
        
        logger.info(f"{metro_code}: Historical YoY change (2002-2018) mean={historical_mean_change:.2f}%, "
                   f"SD={historical_sd_change:.2f}%, "
                   f"Bounds=[{lower_bound:.2f}%, {upper_bound:.2f}%] using {SD_MULTIPLIER} SD")
        
        return lower_bound, upper_bound, historical_mean_change, historical_sd_change, True
    else:
        logger.warning(f"{metro_code}: Insufficient historical YoY data from 2002-2018 ({len(valid_yoy_changes)} periods). "
                      f"Using default bounds.")
        return -5.0, 5.0, 0.0, 2.5, False

def forecast_metro(metro_code: str, 
                  metro_name: str, 
                  model_package: Dict[str, Any], 
                  future_data_all: pd.DataFrame,
                  historical_data: pd.DataFrame = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast for a single metro with enhanced error handling"""
    
    try:
        model_info = model_package['model_info']
        feature_config = model_package['feature_engineering']
        historical_context = model_package['historical_context']
        
        if pd.isna(metro_code) or metro_code == 'NA':
            metro_code = 'NA'
            metro_data = future_data_all[future_data_all['Metro Code'] == 'NA'].copy()
            if len(metro_data) == 0:
                metro_data = future_data_all[future_data_all['Location'].str.contains('Nashville', case=False, na=False)].copy()
        elif metro_code in future_data_all['Metro Code'].unique():
            metro_data = future_data_all[future_data_all['Metro Code'] == metro_code].copy()
        else:
            safe_metro_code = metro_code.replace('NA_Metro', 'NA')
            if safe_metro_code in future_data_all['Metro Code'].unique():
                metro_data = future_data_all[future_data_all['Metro Code'] == safe_metro_code].copy()
            else:
                return None, f"No future data found for metro {metro_code}"
        
        logger.debug(f"Processing {len(metro_data)} records for {metro_code}")
        
        metro_data = create_features_for_metro(metro_data, model_info, feature_config)
        
        # BUG #5: This filtering doesn't consider quarters
        # Current implementation only filters by year
        forecast_data = metro_data[
            (metro_data['year'] >= FORECAST_START_YEAR) & 
            (metro_data['year'] <= FORECAST_END_YEAR)
        ].copy()
        
        # TODO: Update the filtering logic to include quarter boundaries
        # Hint: You need to check:
        # - If year equals start year, month should be >= start quarter month
        # - If year equals end year, month should be <= end quarter month
        # - If year is between start and end, include all quarters
        
        if len(forecast_data) < MIN_FORECAST_PERIODS:
            return None, f"Insufficient forecast periods: {len(forecast_data)} < {MIN_FORECAST_PERIODS}"
        
        features_needed = model_info['features']
        X_future = forecast_data[features_needed].copy()
        
        # Handle NaN values
        nan_counts = X_future.isnull().sum()
        if nan_counts.any():
            logger.warning(f"NaN values detected for {metro_code}: {dict(nan_counts[nan_counts > 0])}")
            
            for col in features_needed:
                if col in X_future.columns and X_future[col].isnull().any():
                    X_future[col] = X_future[col].interpolate(method='linear', limit=2)
                    X_future[col] = X_future[col].ffill(limit=2)
                    X_future[col] = X_future[col].bfill(limit=2)
            
            nan_counts_after = X_future.isnull().sum()
            if nan_counts_after.any():
                logger.warning(f"NaN values remain after interpolation: {dict(nan_counts_after[nan_counts_after > 0])}")
                valid_mask = ~X_future.isnull().any(axis=1)
                X_future = X_future[valid_mask]
                forecast_data = forecast_data[valid_mask]
        
        if len(X_future) == 0:
            return None, "No valid data after feature engineering"
        
        if feature_config.get('scaling_required', False):
            if 'scaler' not in feature_config:
                return None, "Scaler required but not found in model package"
                
            scaler = feature_config['scaler']
            X_future_scaled = pd.DataFrame(
                scaler.transform(X_future),
                columns=X_future.columns,
                index=X_future.index
            )
            X_future = X_future_scaled
            logger.debug(f"Applied {feature_config.get('scaler_type', 'Unknown')} scaling")
        
        intercept = model_info['intercept']
        coefficients = model_info['coefficients']
        
        missing_coefs = [f for f in features_needed if f not in coefficients]
        if missing_coefs:
            logger.warning(f"Missing coefficients for features: {missing_coefs}")
        
        predictions_yoy_change = np.full(len(X_future), intercept)
        for feature in features_needed:
            if feature in coefficients:
                predictions_yoy_change += X_future[feature].values * coefficients[feature]
        
        forecast_results = []
        
        # BUG #6: Getting last known asking rent doesn't consider the start quarter
        last_known_year = FORECAST_START_YEAR - 1
        # TODO: This should get the last known value before the start quarter, not just the start year
        metro_last_known = metro_data[metro_data['year'] == last_known_year]
        if len(metro_last_known) > 0 and 'Asking Rent/SF' in metro_last_known.columns:
            valid_asking_rent = metro_last_known['Asking Rent/SF'].dropna()
            last_asking_rent = valid_asking_rent.iloc[-1] if len(valid_asking_rent) > 0 else DEFAULT_ASKING_RENT
        else:
            last_asking_rent = DEFAULT_ASKING_RENT
            logger.warning(f"Using default asking rent {DEFAULT_ASKING_RENT} $/SF for {metro_code}")
        
        asking_rent_history = {}
        if len(metro_last_known) > 0:
            for _, row in metro_last_known.iterrows():
                if 'Asking Rent/SF' in row and not pd.isna(row.get('Asking Rent/SF')):
                    asking_rent_history[row['date_col']] = row['Asking Rent/SF']
        
        logger.debug(f"Asking rent history has {len(asking_rent_history)} entries for {metro_code}")
        
        if USE_SD_CAPPING:
            data_for_bounds = historical_data if historical_data is not None else metro_data
            lower_bound, upper_bound, hist_mean, hist_sd, has_bounds = calculate_metro_bounds(
                metro_code, data_for_bounds
            )
        else:
            lower_bound, upper_bound = 0, 100
            hist_mean, hist_sd, has_bounds = None, None, False
        
        yoy_change_history = {}
        
        # BUG #7: Special handling for first forecast period needs quarter awareness
        for idx, (i, row) in enumerate(forecast_data.iterrows()):
            current_date = row['date_col']
            prev_year_date = current_date - pd.DateOffset(years=1)
            
            base_rate = None
            
            if prev_year_date in asking_rent_history:
                base_rate = asking_rent_history[prev_year_date]
            else:
                close_dates = [
                    d for d in asking_rent_history.keys() 
                    if abs((d - prev_year_date).days) <= DATE_TOLERANCE_DAYS
                ]
                if close_dates:
                    closest_date = min(close_dates, key=lambda d: abs((d - prev_year_date).days))
                    base_rate = asking_rent_history[closest_date]
                    days_diff = abs((closest_date - prev_year_date).days)
                    logger.debug(f"Using base rate from {days_diff} days away for {current_date}")
                else:
                    base_rate = last_asking_rent
            
            raw_yoy_change = predictions_yoy_change[idx]
            
            if USE_SD_CAPPING and has_bounds:
                capped_yoy_change = np.clip(raw_yoy_change, lower_bound, upper_bound)
                was_capped = (capped_yoy_change != raw_yoy_change)
            else:
                capped_yoy_change = np.clip(raw_yoy_change, -20.0, 20.0)
                was_capped = (capped_yoy_change != raw_yoy_change)
            
            yoy_change_history[current_date] = capped_yoy_change
            
            # BUG #8: This special handling needs to consider the start quarter
            if row['year'] == FORECAST_START_YEAR:
                # TODO: This logic should check if we're in the first forecast quarter
                # not just the first forecast year
                quarter = row['month']
                
                # The rest of the special handling logic...
                predicted_asking_rent = base_rate * (1 + capped_yoy_change / 100)
            else:
                predicted_asking_rent = base_rate * (1 + capped_yoy_change / 100)
            
            predicted_asking_rent = max(predicted_asking_rent, 0.01)
            asking_rent_history[current_date] = predicted_asking_rent
            
            forecast_results.append({
                'Date': current_date,
                'Metro_Code': metro_code,
                'Metro_Name': metro_name,
                'Scenario': SCENARIO_NAME,
                'Feature_Set': model_info['name'],
                'Model_Type': model_info['type'],
                'Predicted_Asking_Rent': predicted_asking_rent,
                'Base_Asking_Rent': base_rate,
                'Raw_Predicted_Change_Pct': raw_yoy_change,
                'Capped_Predicted_Change_Pct': capped_yoy_change,
                'Was_YoY_Capped': was_capped,
                'Cap_Type': 'SD_YoY' if USE_SD_CAPPING and has_bounds else 'Default_YoY',
                'YoY_Lower_Bound': lower_bound,
                'YoY_Upper_Bound': upper_bound,
                'Previous_Year_Date': prev_year_date,
                'PropertyType': 'Warehouse',
                'Metric': 'Asking Rent',
                'Model_Tier': model_package['metadata'].get('tier_used', 'Unknown'),
                'F1_Score': model_info['performance']['F1_Score'],
                'RMSE': model_info['performance']['RMSE'],
                'Directional_Accuracy': model_info['performance'].get('Directional_Accuracy', None),
                'Historical_YoY_Mean': hist_mean if has_bounds else None,
                'Historical_YoY_SD': hist_sd if has_bounds else None
            })
        
        return pd.DataFrame(forecast_results), None
        
    except Exception as e:
        logger.error(f"Error forecasting {metro_code}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"Exception: {str(e)}"

def main():
    start_time = time.time()
    
    # BUG #9: Configuration display needs to show quarters
    print(f"\n📊 CONFIGURATION:")
    print(f"   • Scenario: {SCENARIO_NAME}")
    print(f"   • Future Economic Data: {FUTURE_ECONOMIC_DATA_FILE}")
    print(f"   • Historical Data: {HISTORICAL_DATA_FILE}")
    print(f"   • Capping Method: {'SD-based YoY' if USE_SD_CAPPING else 'Default YoY (±20%)'}")
    # TODO: Add display of quarter configuration here
    if USE_SD_CAPPING:
        print(f"   • SD Multiplier: {SD_MULTIPLIER} (±{SD_MULTIPLIER} standard deviations on YoY changes)")
        print(f"   • Min Historical Periods: {MIN_HISTORICAL_PERIODS}")
    print()
    
    latest_folder = find_latest_results_folder()
    
    if not latest_folder:
        print("❌ No multi-metro results folder found!")
        print("   Please run 'run_multi_metro_safe_updated.py' first")
        return
    
    print(f"📁 Found latest results folder: {latest_folder}")
    
    results_file = os.path.join(latest_folder, 'processing_results.csv')
    if not os.path.exists(results_file):
        logger.error(f"Processing results not found at {results_file}")
        return
    
    processing_results = pd.read_csv(results_file, keep_default_na=False, na_values=[''])
    
    if 'metro_code' in processing_results.columns:
        mask = (processing_results['metro_code'].isna()) & (processing_results['metro_name'] == 'Nashville')
        processing_results.loc[mask, 'metro_code'] = 'NA'
    
    successful_metros = processing_results[processing_results['status'] == 'success']
    
    print(f"\n📊 Found {len(successful_metros)} successfully processed metros")
    
    print("\n📈 Loading future economic data...")
    try:
        future_data = pd.read_csv(FUTURE_ECONOMIC_DATA_FILE, keep_default_na=False, na_values=[''])
        future_data['date_col'] = pd.to_datetime(future_data['date_col'], format='%d/%m/%Y', errors='coerce')
        
        if future_data['date_col'].isna().any():
            future_data['date_col'] = pd.to_datetime(future_data['date_col'], format='%m/%d/%Y', errors='coerce')
        
        numeric_cols = ['10-Yr Bond Yield', 'Consumer price index', 
                       'Employment - Utilities, transportation and warehousing',
                       'GDP, real - Transportation and warehousing',
                       'Retail sales (excluding sales of vehicles), real',
                       'Consumer spending, real']
        for col in numeric_cols:
            if col in future_data.columns and future_data[col].dtype == 'object':
                future_data[col] = future_data[col].astype(str).str.replace(',', '')
                future_data[col] = pd.to_numeric(future_data[col], errors='coerce')
        
        future_data['month'] = future_data['date_col'].dt.month
        future_data['year'] = future_data['date_col'].dt.year
        future_data = future_data.sort_values('date_col')
        print(f"✅ Loaded {len(future_data)} records of future data")
        
        required_cols = ['Metro Code', 'date_col', 'year', 'month', '10-Yr Bond Yield', 
                        'Consumer price index', 'Employment - Utilities, transportation and warehousing',
                        'GDP, real - Transportation and warehousing',
                        'Retail sales (excluding sales of vehicles), real',
                        'Consumer spending, real']
        missing = [col for col in required_cols if col not in future_data.columns]
        if missing:
            raise ValueError(f"Missing required columns in future data: {missing}")
        
        if 'Vacancy_Ratio' in future_data.columns:
            vacancy_ratio_count = future_data['Vacancy_Ratio'].notna().sum()
            logger.info(f"✅ Vacancy_Ratio column found in data with {vacancy_ratio_count} non-null values")
            logger.info(f"   Vacancy_Ratio range: {future_data['Vacancy_Ratio'].min():.3f} to {future_data['Vacancy_Ratio'].max():.3f}")
        else:
            logger.info("ℹ️ Vacancy_Ratio column not found in data - will be calculated using HP filter")
            
    except Exception as e:
        logger.error(f"Error loading future data: {e}")
        return
    
    historical_data = None
    if USE_SD_CAPPING:
        print("\n📊 Loading historical data for SD calculation...")
        try:
            if os.path.exists(HISTORICAL_DATA_FILE):
                historical_data = pd.read_csv(HISTORICAL_DATA_FILE, keep_default_na=False, na_values=[''])
                historical_data['date_col'] = pd.to_datetime(historical_data['date_col'], format='%d/%m/%Y', errors='coerce')
                
                if historical_data['date_col'].isna().any():
                    historical_data['date_col'] = pd.to_datetime(historical_data['date_col'], format='%m/%d/%Y', errors='coerce')
                
                numeric_cols = ['10-Yr Bond Yield', 'Consumer price index', 
                               'Employment - Utilities, transportation and warehousing',
                               'GDP, real - Transportation and warehousing',
                               'Retail sales (excluding sales of vehicles), real',
                               'Consumer spending, real']
                for col in numeric_cols:
                    if col in historical_data.columns and historical_data[col].dtype == 'object':
                        historical_data[col] = historical_data[col].astype(str).str.replace(',', '')
                        historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')
                
                historical_data['year'] = historical_data['date_col'].dt.year
                print(f"✅ Loaded {len(historical_data)} records of historical data")
            else:
                logger.warning(f"{HISTORICAL_DATA_FILE} not found. Will use limited data for SD bounds.")
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}. Will use limited data for SD bounds.")
    
    all_forecasts = []
    success_count = 0
    failed_metros = []
    
    print(f"\n🚀 Generating forecasts for {len(successful_metros)} metros...")
    print("=" * 70)
    
    for idx, metro_row in successful_metros.iterrows():
        metro_start = time.time()
        metro_code = metro_row['metro_code']
        metro_name = metro_row['metro_name']
        
        if idx > 0:
            avg_time = (time.time() - start_time) / idx
            eta_seconds = avg_time * (len(successful_metros) - idx)
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            print(f"\n[{idx+1}/{len(successful_metros)}] {metro_code} - {metro_name} (ETA: {eta.strftime('%H:%M:%S')})")
        else:
            print(f"\n[{idx+1}/{len(successful_metros)}] {metro_code} - {metro_name}")
        
        if pd.isna(metro_code) and metro_name == 'Nashville':
            metro_code = 'NA'
            folder_code = 'NA_Metro'
        else:
            folder_code = metro_code.replace('NA', 'NA_Metro') if metro_code == 'NA' else metro_code
        
        metro_folder = os.path.join(latest_folder, f"{folder_code}_{metro_name.replace(' ', '_')}")
        
        model_package, error = load_metro_model(metro_folder)
        if error:
            print(f"   ❌ {error}")
            failed_metros.append({'metro_code': metro_code, 'metro_name': metro_name, 'error': error})
            continue
        
        try:
            forecast_df, error = forecast_metro(metro_code, metro_name, model_package, future_data, historical_data)
            
            if error:
                print(f"   ❌ {error}")
                failed_metros.append({'metro_code': metro_code, 'metro_name': metro_name, 'error': error})
                continue
            
            if forecast_df is not None and len(forecast_df) > 0:
                all_forecasts.append(forecast_df)
                success_count += 1
                avg_asking_rent = forecast_df['Predicted_Asking_Rent'].mean()
                metro_time = time.time() - metro_start
                print(f"   ✅ Generated {len(forecast_df)} forecasts | Avg asking rent: ${avg_asking_rent:.2f}/SF | Time: {metro_time:.1f}s")
            else:
                print(f"   ❌ No forecasts generated")
                failed_metros.append({'metro_code': metro_code, 'metro_name': metro_name, 'error': 'No forecasts generated'})
                
        except Exception as e:
            logger.error(f"Exception processing {metro_code}: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
            failed_metros.append({'metro_code': metro_code, 'metro_name': metro_name, 'error': str(e)})
    
    if all_forecasts:
        final_forecasts = pd.concat(all_forecasts, ignore_index=True)
        
        logger.info(f"Final dataset shape: {final_forecasts.shape}")
        logger.info(f"Date range: {final_forecasts['Date'].min()} to {final_forecasts['Date'].max()}")
        logger.info(f"Unique metros: {final_forecasts['Metro_Code'].nunique()}")
        
        capped_count = final_forecasts['Was_YoY_Capped'].sum()
        capped_pct = (capped_count / len(final_forecasts)) * 100
        print(f"\n📊 YoY CAPPING SUMMARY:")
        print(f"   • Total YoY changes capped: {capped_count:,} ({capped_pct:.1f}%)")
        
        numeric_cols = ['Predicted_Asking_Rent', 'Base_Asking_Rent', 
                       'Raw_Predicted_Change_Pct', 'Capped_Predicted_Change_Pct', 
                       'F1_Score', 'RMSE', 'Directional_Accuracy', 
                       'YoY_Lower_Bound', 'YoY_Upper_Bound', 
                       'Historical_YoY_Mean', 'Historical_YoY_SD']
        for col in numeric_cols:
            if col in final_forecasts.columns:
                final_forecasts[col] = final_forecasts[col].round(3)
        
        final_forecasts = final_forecasts.sort_values(['Metro_Code', 'Date'])
        
        print("\n📈 Applying HP filter to smooth forecasts...")
        hp_smoothed_values = []
        
        for metro_code in final_forecasts['Metro_Code'].unique():
            metro_data = final_forecasts[final_forecasts['Metro_Code'] == metro_code].copy()
            asking_rent_series = metro_data['Predicted_Asking_Rent'].values
            
            try:
                cycle, trend = hpfilter(asking_rent_series, lamb=20)
                hp_smoothed_values.extend(trend)
            except Exception as e:
                logger.warning(f"HP filter failed for {metro_code}: {e}. Using original values.")
                hp_smoothed_values.extend(asking_rent_series)
        
        final_forecasts['Predicted_Asking_Rent_HP20'] = hp_smoothed_values
        final_forecasts['Predicted_Asking_Rent_HP20'] = final_forecasts['Predicted_Asking_Rent_HP20'].round(3)
        
        print(f"✅ HP filtering complete for {final_forecasts['Metro_Code'].nunique()} metros")
        
        # BUG #10: Output filename should include quarter information
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        output_folder = f'centralized_forecasts_{timestamp}'
        os.makedirs(output_folder, exist_ok=True)
        
        # TODO: Update filename to include quarters
        csv_filename = f'all_metro_forecasts_{FORECAST_START_YEAR}_{FORECAST_END_YEAR}.csv'
        # Should be something like: f'all_metro_forecasts_{FORECAST_START_YEAR}_Q{FORECAST_START_QUARTER}_{FORECAST_END_YEAR}_Q{FORECAST_END_QUARTER}.csv'
        
        csv_path = os.path.join(output_folder, csv_filename)
        final_forecasts.to_csv(csv_path, index=False)
        
        print(f"\n✅ Forecast generation complete!")
        print(f"   • Successfully forecasted: {success_count}/{len(successful_metros)} metros")
        print(f"   • Total forecast records: {len(final_forecasts):,}")
        print(f"   • Total runtime: {(time.time() - start_time) / 60:.1f} minutes")
        print(f"\n📁 Results saved to: {output_folder}/")
        print(f"   • Detailed CSV: {csv_filename}")
        print(f"   • Log file: {log_filename}")
        
    else:
        print("\n❌ No forecasts were generated successfully")
        logger.error("No successful forecasts generated")
    
    print("\n" + "=" * 70)
    print("🎉 Centralized forecasting complete!")

if __name__ == "__main__":
    main()