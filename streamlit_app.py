import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pyRiskadjusted
import numpy as np
from fmp_python.fmp import FMP

fmp = FMP(api_key=st.secrets['API_key'], output_format='pandas')

def compute_portfolio_metrics(weights, tickers):
    
    def import_data_from_FMP(tickers):
        data = {}
        for ticker in tickers:
            df = fmp.get_historical_price(ticker)
            df = df.iloc[::-1].reset_index(drop=True)  # Reverse the order
            data[ticker] = df
        return data

    def clean_and_transform_data(portfolio_data):
        transformed_data = {}
        for ticker, df in portfolio_data.items():
            cleaned_df = df[['date', 'adjClose']].copy()
            cleaned_df.rename(columns={'date': 'Date', 'adjClose': 'Price'}, inplace=True)
            cleaned_df["Return"] = cleaned_df["Price"].pct_change(periods=1)
            cleaned_df.drop(index=cleaned_df.index[0], axis=0, inplace=True)
            cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"])
            cleaned_df.set_index(keys="Date", drop=True, inplace=True)
            cleaned_df = cleaned_df.apply(lambda x: x.astype(float))
            transformed_data[ticker] = cleaned_df
        return transformed_data

    # Importing and cleaning the data
    data = import_data_from_FMP(tickers)
    data = clean_and_transform_data(data)

    # Constructing the portfolio DataFrame
    portfolio_df = pd.concat([data[ticker]['Return'] for ticker in tickers], axis=1)
    portfolio_df.columns = tickers

    # Covariance matrix
    covariance_matrix = portfolio_df[tickers].cov()

    # Standard Deviation (Std)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    # Returns
    portfolio_returns = (portfolio_df * weights).sum(axis=1)

    # Mean
    avg_rets = portfolio_df[tickers].mean()
    portfolio_mean = avg_rets.dot(weights)

    return portfolio_std_dev, portfolio_mean, portfolio_returns

def calculate_var_cvar(alpha, portfolio_returns, portfolio_mean, portfolio_std_dev):
    df = pd.DataFrame(columns=['Method', 'VaR', 'CVaR'])
    
    # Dictionary to store the figures for each method
    figures = {}
    
    methods = {
        "Empirical": {
            "var": pyRiskadjusted.ValueAtRisk(array=portfolio_returns, alpha=alpha, axis=0).empirical_var(plot=False, bins=50),
            "es": pyRiskadjusted.ExpectedShortfall(array=portfolio_returns, alpha=alpha, axis=0).empirical_cvar(plot=False, bins=50)
        },
        "Parametrical": {
            "var": pyRiskadjusted.ValueAtRisk(array=portfolio_returns, alpha=alpha, axis=0, portfolio_mean=portfolio_mean, portfolio_std_dev=portfolio_std_dev).parametrical_var(plot=False, bins=50),
            "es": pyRiskadjusted.ExpectedShortfall(array=portfolio_returns, alpha=alpha, axis=0, portfolio_mean=portfolio_mean, portfolio_std_dev=portfolio_std_dev).parametrical_cvar(plot=False, bins=50)
        },
        "Non Parametrical": {
            "var": pyRiskadjusted.ValueAtRisk(array=portfolio_returns, alpha=alpha, axis=0).non_parametrical_var(random_state=42, n_iter=100000, plot=False, bins=50),
            "es": pyRiskadjusted.ExpectedShortfall(array=portfolio_returns, alpha=alpha, axis=0).non_parametrical_cvar(random_state=42, n_iter=100000, plot=False, bins=50)
        },
        "EVT": {
            "var": pyRiskadjusted.ValueAtRisk(array=portfolio_returns, alpha=alpha, axis=0).extreme_var(k=8, plot=False, bins=50),
            "es": pyRiskadjusted.ExpectedShortfall(array=portfolio_returns, alpha=alpha, axis=0).extreme_cvar(k=8, plot=True, bins=50)
        }
    }
    
    # Save the figures for each method
    figures["Empirical"] = pyRiskadjusted.ExpectedShortfall(array=portfolio_returns, alpha=alpha, axis=0).empirical_cvar(plot=True, bins=50)[1]
    figures["Parametrical"] = pyRiskadjusted.ExpectedShortfall(array=portfolio_returns, alpha=alpha, axis=0, portfolio_mean=portfolio_mean, portfolio_std_dev=portfolio_std_dev).parametrical_cvar(plot=True, bins=50)[1]
    figures["Non Parametrical"] = pyRiskadjusted.ExpectedShortfall(array=portfolio_returns, alpha=alpha, axis=0).non_parametrical_cvar(random_state=42, n_iter=100000, plot=True, bins=50)[1]
    figures["EVT"] = pyRiskadjusted.ExpectedShortfall(array=portfolio_returns, alpha=alpha, axis=0).extreme_cvar(k=8, plot=True, bins=50)[1]
    
    empirical_var = methods["Empirical"]["var"]
    parametrical_var = methods["Parametrical"]["var"]
    non_parametrical_var = methods["Non Parametrical"]["var"]
    evt_var = methods["EVT"]["var"]
    
    for method_name, method_funcs in methods.items():
        var_value = method_funcs["var"]
        es_value = method_funcs["es"]
        if isinstance(var_value, tuple):
            var_value = var_value[0]
        if isinstance(es_value, tuple):
            es_value = es_value[0]
        
        new_row = {
            "Method": method_name,
            "VaR": var_value,
            "CVaR": es_value
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df, empirical_var, parametrical_var, non_parametrical_var, evt_var, figures

def backtest_var_methods(portfolio_returns, empirical_var, parametrical_var, non_parametrical_var, extreme_var):

    # Splitting the dataset
    train = portfolio_returns.iloc[:round(portfolio_returns.shape[0]*0.5)]
    test = portfolio_returns.iloc[round(portfolio_returns.shape[0]*0.5):]

    # Define a helper function to create a DataFrame row
    def create_row(test_name, statistic, p_value=None, quantile=None, chi_square=None, decision=None):
        return pd.DataFrame({
            "Test": [test_name],
            "Statistic": [statistic],
            "P-value": [p_value],
            "Quantile": [quantile],
            "Chi-square": [chi_square],
            "Decision": [decision]
        })

    # Dictionary to store VaR methods and their values
    var_methods = {
        "Empirical": empirical_var,
        "Parametrical": parametrical_var,
        "Non-Parametrical": non_parametrical_var,
        "Extreme": extreme_var
    }

    # Create a dictionary to store results DataFrames for each method
    results_dfs = {}

    # Iterate over each VaR method and run backtests
    for method_name, method_value in var_methods.items():

        # Initialize an empty list to store results for this method
        results_list = []

        # Student test
        student = pyRiskadjusted.BackTesting(array=test.copy(), axis=0).student_test(threshold=method_value, alpha=0.05)
        results_list.append(create_row("Student test", student["Statistic"], p_value=student["P-value"], decision=student["Decision"]))

        # Normal test
        normal = pyRiskadjusted.BackTesting(array=test.copy(), axis=0).normal_test(threshold=method_value, alpha=0.05)
        results_list.append(create_row("Normal test", normal["Statistic"], quantile=normal["Quantile"], decision=normal["Decision"]))

        # Kupiec test
        kupiec = pyRiskadjusted.BackTesting(array=test.copy(), axis=0).kupiec_test(threshold=method_value, alpha=0.01)
        results_list.append(create_row("Kupiec test", kupiec["Statistic"], chi_square=kupiec["Chi-square"], decision=kupiec["Decision"]))

        # Christoffersen test
        christoffersen = pyRiskadjusted.BackTesting(array=test.copy(), axis=0).christoffersen_test(threshold=method_value, alpha=0.01)
        results_list.append(create_row("Christoffersen test", christoffersen["Statistic"], chi_square=christoffersen["Chi-square"], decision=christoffersen["Decision"]))

        # Kupiec & Christoffersen test
        kupiec_christoffersen = pyRiskadjusted.BackTesting(array=test.copy(), axis=0).kupiec_christoffersen_test(threshold=method_value, alpha=0.01)
        results_list.append(create_row("Kupiec & Christoffersen test", kupiec_christoffersen["Statistic"], chi_square=kupiec_christoffersen["Chi-square"], decision=kupiec_christoffersen["Decision"]))

        # Concatenate the results into a DataFrame for this method and add to the dictionary
        results_dfs[method_name] = pd.concat(results_list, ignore_index=True)

    return results_dfs

def combined_function(portfolio_returns, alpha, portfolio_mean=None, portfolio_std_dev=None):
    # 1. PickandsEstimator
    pickands_result_value, pickands_fig = pyRiskadjusted.PickandsEstimator(
        array=portfolio_returns,
        alpha=alpha,
        k=8,
        axis=0
    ).gev_parameter(plot=True, n_iter=100)
    
    pickands_result = {
        "Pickands Result": pickands_result_value,
        "Figure": pickands_fig
    }
    
    # 2. Leadbetter
    leadbetter_result_value = pyRiskadjusted.Leadbetter(
        array=portfolio_returns,
        threshold=pyRiskadjusted.ValueAtRisk(
            array=portfolio_returns,
            alpha=alpha,
            axis=0
        ).empirical_var(plot=False),
        axis=0
    ).extremal_index()
    
    leadbetter_result = {
        "Leadbetter Result (Empirical)": leadbetter_result_value
    }
    
    # 3. BackTesting
    backtesting_result = pyRiskadjusted.BackTesting(
        array=portfolio_returns,
        axis=0
    ).var_diameter(
        var=[
            pyRiskadjusted.ValueAtRisk(array=portfolio_returns, alpha=alpha, axis=0).empirical_var(plot=False),
            pyRiskadjusted.ValueAtRisk(array=portfolio_returns, alpha=alpha, axis=0, portfolio_mean=portfolio_mean, portfolio_std_dev=portfolio_std_dev).parametrical_var(plot=False),
            pyRiskadjusted.ValueAtRisk(array=portfolio_returns, alpha=alpha, axis=0).non_parametrical_var(random_state=42, n_iter=100000, plot=False)[0],
            pyRiskadjusted.ValueAtRisk(array=portfolio_returns, alpha=alpha, axis=0).extreme_var(k=8, plot=False)
        ],
        alpha=alpha
    )
    
    return pickands_result, leadbetter_result, backtesting_result

if "portfolio_returns" not in st.session_state:
    st.session_state.portfolio_returns = None

if "empirical_var" not in st.session_state:
    st.session_state.empirical_var = None

st.title("Risk-VaR")

# User input for tickers
ticker_input = st.text_input("Enter Tickers (comma-separated)", "AAPL,MSFT,AMZN,GOOGL,META")
tickers = [ticker.strip() for ticker in ticker_input.split(',')]

# User input for alpha
alpha = st.number_input(
    "Input Alpha (confidence level)",
    min_value=0.001,
    max_value=0.1,
    value=0.01,
    step=0.001,
    format="%.3f"
)

# User input for weights
weights_input = st.text_input("Enter Weights (comma-separated)", "0.2,0.2,0.2,0.2,0.2")
weights_list = [float(w.strip()) for w in weights_input.split(',')]

# Normalize the weights
total_weight = sum(weights_list)
weights = np.array(weights_list) / total_weight

# Compute portfolio metrics button
compute_metrics_button = st.button("Compute Portfolio Metrics")

if compute_metrics_button:
    st.session_state.portfolio_std_dev, st.session_state.portfolio_mean, st.session_state.portfolio_returns = compute_portfolio_metrics(weights, tickers)
    st.write("Portfolio Standard Deviation:", st.session_state.portfolio_std_dev)
    st.write("Portfolio Mean:", st.session_state.portfolio_mean)

run_methods_button = st.button("Run Methods")

if run_methods_button:
    if st.session_state.portfolio_returns is None:
        st.warning("Please compute portfolio metrics first.")
    else:
        df_result, st.session_state.empirical_var, st.session_state.parametrical_var, st.session_state.non_parametrical_var, st.session_state.extreme_var, figs = calculate_var_cvar(alpha, st.session_state.portfolio_returns, st.session_state.portfolio_mean, st.session_state.portfolio_std_dev)
        
        # Display the dataframe
        st.write(df_result)
        
        # Display the plots
        for method, fig in figs.items():
            st.write(f"{method}")
            st.pyplot(fig)

run_backtests_button = st.button("Run Backtests")

if run_backtests_button:
    if st.session_state.empirical_var is None:
        st.warning("Please compute portfolio metrics first.")
    else:
        st.session_state.non_parametrical_var=st.session_state.non_parametrical_var[0]
        dfs = backtest_var_methods(st.session_state.portfolio_returns, st.session_state.empirical_var, st.session_state.parametrical_var, st.session_state.non_parametrical_var, st.session_state.extreme_var)
        for key, df in dfs.items():
            st.write(f"Results for {key} VaR Method:")
            st.write(df)

        pickands_result, leadbetter_result, backtesting_result = combined_function(st.session_state.portfolio_returns, alpha, st.session_state.portfolio_mean, st.session_state.portfolio_std_dev)


        st.pyplot(pickands_result["Figure"])

        pickands_df = pd.DataFrame([pickands_result["Pickands Result"]], columns=["Pickands Result"])
        st.dataframe(pickands_df)

        # Display the Leadbetter result as a DataFrame
        leadbetter_df = pd.DataFrame([leadbetter_result["Leadbetter Result (Empirical)"]], columns=["Leadbetter Result (Empirical)"])
        st.dataframe(leadbetter_df)

        # Display the Backtesting result as a DataFrame
        diameter_data = {"Diameter for VaR models": [backtesting_result["Diameter for VaR models"]]}

        # Convert the extracted data into a DataFrame
        backtesting_df = pd.DataFrame(diameter_data)

        # Display the DataFrame in Streamlit
        st.dataframe(backtesting_df)