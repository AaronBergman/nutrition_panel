# Import necessary libraries
import streamlit as st
import pandas as pd
from linearmodels import PanelOLS
from linearmodels.panel import PooledOLS

# Load the CSV data
data = pd.read_csv("cdp.csv")
data = data.set_index(["country_name", "year"])  # Setting a multi-index for panel data

# Create a list of all columns (excluding "country_name" and "year") for the dropdown menu
columns_for_dropdown = [col for col in data.columns if col not in ["country_code", "adult_obesity_pct"]]

# Define the Streamlit app
def main():
    st.title("Linear Panel Regression App")
    st.write("Select the regressor(s) from the dropdown menu below:")

    # Create a multiselect dropdown for selecting the regressor(s)
    selected_columns = st.multiselect("Select Regressor(s)", columns_for_dropdown)

    if selected_columns:  # If some columns are selected

        # Filter the data to include only the selected regressors and the outcome variable
        X = data[selected_columns]
        y = data["adult_obesity_pct"]

        # Run the panel regression using PanelOLS
        model = PanelOLS(y, X, entity_effects=True)
        results = model.fit()

        # Display the summary of the regression results
        st.subheader("Regression Results")
        st.write(results)
    else:  # If no columns are selected
        st.warning("Please select at least one regressor to run the regression.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
