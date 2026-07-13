import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os 

def train_inverse_model(data_file='hhg_inverse_data.csv', model_output_file='model/hhg_inverse_model.pkl'):
    """
    Loads the generated data, trains a Random Forest Regressor
    to predict parameters from spectrum features.
    """
    
    # 1. Load Data
    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
        print("Please run `python data_generator.py` first to create the dataset.")
        return
        
    if data.empty or len(data) < 10:
        print(f"Error: Data file '{data_file}' is empty or has too few samples. Need at least 10.")
        print("Please run `python data_generator.py` with more samples.")
        return

    print(f"Loaded {len(data)} samples from {data_file}")
    
    # Drop any rows with NaN/Inf values that might have slipped through
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    print(f"Using {len(data)} valid samples after cleaning.")


    # 2. Define Features (X) and Targets (y)
    # The inputs to our model (from the spectrum)
    features = ['cutoff_order', 'plateau_intensity_mean', 'H5_intensity', 'H11_intensity', 'H21_intensity']
    # The outputs we want to predict (the physics parameters)
    targets = ['E0', 'omega0']
    
    X = data[features]
    y = data[targets]

    # 3. Create Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples.")

    # 4. Initialize and Train the Model
    # RandomForestRegressor natively supports multi-output regression
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1 
    )
    
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the Model
    y_pred = model.predict(X_test)
    
    # Calculate metrics for each target individually
    print("\n\t\t--- Model Evaluation ---")
    for i, target_name in enumerate(targets):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"  Target: {target_name}")
        print(f"    Mean Squared Error (MSE): {mse:.6f}")
        print(f"    R-squared (R2) Score:   {r2:.3f} (closer to 1.0 is better)")
    # 6. Save the Trained Model
    joblib.dump(model, model_output_file)
    print(f"\nSuccessfully trained and saved inverse model to {model_output_file}")

    # --- Plotting Results ---
    
    # Plot Feature Importances
    # For multi-output, importances are averaged across all outputs
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importances for Predicting Parameters')
    plt.xlabel('Importance')
    plt.ylabel('Spectrum Feature')
    plt.tight_layout()
    plt.savefig('inverse_feature_importances.png')
    print("Saved feature importance plot to 'inverse_feature_importances.png'")

    # Plot Prediction vs Actual for each target
    for i, target_name in enumerate(targets):
        plt.figure(figsize=(10, 6))
        
        # Get the actual and predicted values for this target
        y_test_target = y_test.iloc[:, i]
        y_pred_target = y_pred[:, i]
        
        plt.scatter(y_test_target, y_pred_target, alpha=0.5, label='Predicted')
        
        # Plot the "ideal" line
        min_val = min(y_test_target.min(), y_pred_target.min())
        max_val = max(y_test_target.max(), y_pred_target.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (Actual)')
        
        plt.title(f'Model Predictions vs. Actual Values for {target_name}')
        plt.xlabel(f'Actual {target_name}')
        plt.ylabel(f'Predicted {target_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_filename = f'results/predictions_vs_actual_{target_name}.png'
        plt.savefig(plot_filename)
        print(f"Saved prediction plot to '{plot_filename}'")


if __name__ == "__main__":
    train_inverse_model()