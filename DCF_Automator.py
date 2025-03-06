import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI

def fetch_financials(ticker):
    """
    Fetches financial data via yfinance.
    """
    stock = yf.Ticker(ticker)
    income_stmt = stock.financials
    cash_flow_stmt = stock.cashflow
    
    # Check if the data is empty
    if income_stmt.empty or cash_flow_stmt.empty:
        print(f"No financial data available for ticker: {ticker}")
        return None, None
    
    # Print available fields to help with debugging
    print("Income Statement Fields:")
    print(income_stmt.index)
    print("\nCash Flow Fields:")
    print(cash_flow_stmt.index)
    
    return income_stmt, cash_flow_stmt

def calculate_metrics(income_stmt, cash_flow_stmt):
    """
    Calculate key financial metrics and margins
    """
    try:
        # Check if DataFrames are empty
        if income_stmt.empty:
            raise ValueError("Income statement data is empty.")
        if cash_flow_stmt.empty:
            raise ValueError("Cash flow statement data is empty.")

        # Print the entire DataFrame to inspect its structure
        print("Income Statement DataFrame:")
        print(income_stmt)
        print("\nCash Flow Statement DataFrame:")
        print(cash_flow_stmt)

        # Define possible variations of key names
        revenue_keys = ['Total Revenue', 'Revenue', 'Net Revenue', 'Operating Revenue']
        gross_profit_keys = ['Gross Profit', 'Gross Income']
        ebit_keys = ['EBIT', 'Operating Income']
        da_keys = ['Depreciation Amortization Depletion', 'Depreciation And Amortization', 'Depreciation & Amortization', 'Depreciation']
        ocf_keys = ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities']
        capex_keys = ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of PPE']

        # Helper function to find the first available key
        def find_key(keys, df):
            for key in keys:
                if key in df.index:
                    return df.loc[key]
            raise KeyError(f"None of the keys {keys} found in DataFrame.")

        # Use the helper function to find the correct keys
        revenue = find_key(revenue_keys, income_stmt)
        gross_profit = find_key(gross_profit_keys, income_stmt)
        ebit = find_key(ebit_keys, income_stmt)
        da = find_key(da_keys, cash_flow_stmt)
        ebitda = ebit + abs(da)
        ocf = find_key(ocf_keys, cash_flow_stmt)
        capex = find_key(capex_keys, cash_flow_stmt)
        fcf = ocf + capex

        gross_margin = (gross_profit / revenue)
        ebit_margin = (ebit / revenue)
        ebitda_margin = (ebitda / revenue)
        fcf_margin = (fcf / revenue)

        metrics_dict = {
            'Revenue': revenue / 1e6,
            'Gross Profit': gross_profit / 1e6,
            'EBIT': ebit / 1e6,
            'EBITDA': ebitda / 1e6,
            'FCF': fcf / 1e6,
            'Gross Margin %': gross_margin,
            'EBIT Margin %': ebit_margin,
            'EBITDA Margin %': ebitda_margin,
            'FCF Margin %': fcf_margin
        }
        
        return pd.DataFrame(metrics_dict, index=income_stmt.columns)

    except KeyError as e:
        print(f"KeyError: {str(e)} - The key might not exist in the DataFrame.")
        print("Available Income Statement Fields:", income_stmt.index)
        print("Available Cash Flow Fields:", cash_flow_stmt.index)
        raise

    except ValueError as e:
        print(f"ValueError: {str(e)}")
        raise

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        raise

def get_ticker_input():
    """
    Get ticker symbol from user input
    """
    return input("Please enter the ticker symbol (e.g., AAPL): ").upper()

def generate_assumptions_with_openai(historical_metrics, forecast_years):
    """
    Use OpenAI API to generate growth assumptions based on historical metrics
    """
    
    client = OpenAI(api_key=)

    prompt = f"""
    Based on the following historical financial metrics: {historical_metrics.to_dict()},
    please provide year-by-year growth estimates for a {forecast_years} year forecast period.
    For each metric, provide {forecast_years} yearly values (one for each year) in percentage format:

    revenue_growth_y1: [X]%
    revenue_growth_y2: [X]%
    revenue_growth_y3: [X]%
    revenue_growth_y4: [X]%
    revenue_growth_y5: [X]%
    
    gross_margin_y1: [X]%
    ... (continue for all years)
    
    ebit_margin_y1: [X]%
    ... (continue for all years)
    
    da_percent_y1: [X]%
    ... (continue for all years)
    
    fcf_margin_y1: [X]%
    ... (continue for all years)

    Also provide:
    wacc: [X]%
    terminal_growth: [X]%

    Replace [X] with numerical estimates. Provide only the values, no additional text.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Provide only numerical values in the specified format without any explanation."},
            {"role": "user", "content": prompt}
        ]
    )

    response_text = response.choices[0].message.content
    return parse_openai_response(response_text)

def parse_openai_response(response_text):
    """
    Parse the response from OpenAI to extract year-by-year assumptions
    """
    assumptions = {
        'revenue_growth': [],
        'gross_margin': [],
        'ebit_margin': [],
        'da_percent': [],
        'fcf_margin': [],
        'wacc': 0.0,
        'terminal_growth': 0.0
    }
    
    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = float(''.join(c for c in value if c.isdigit() or c in ['.', '-'])) / 100
                
                if 'wacc' in key:
                    assumptions['wacc'] = value
                elif 'terminal_growth' in key:
                    assumptions['terminal_growth'] = value
                else:
                    # Extract the metric type and year
                    for metric in ['revenue_growth', 'gross_margin', 'ebit_margin', 'da_percent', 'fcf_margin']:
                        if metric in key:
                            assumptions[metric].append(value)
                            break
    
    except Exception as e:
        print(f"Error parsing response: {response_text}")
        raise
    
    return assumptions

def calculate_historical_fcf(cash_flow_stmt):
    """
    Extract historical FCF from the statement of cash flows.
    """
    # Print available fields to see what we're working with
    print("Available cash flow statement fields:")
    print(cash_flow_stmt.index)
    
    # Try to find the operating cash flow field (might be named differently)
    try:
        # Common variations of the field names
        ocf_names = [
            'Total Cash From Operating Activities',
            'Operating Cash Flow',
            'Cash Flow From Operations',
            'Net Cash Provided By Operating Activities'
        ]
        
        for name in ocf_names:
            if name in cash_flow_stmt.index:
                ocf = cash_flow_stmt.loc[name]
                break
        else:
            raise KeyError("Could not find operating cash flow field")
            
        # Similar approach for CapEx
        capex_names = [
            'Capital Expenditures',
            'CapEx',
            'Purchase Of Plant And Equipment',
            'Capital Expenditure'
        ]
        
        for name in capex_names:
            if name in cash_flow_stmt.index:
                capex = cash_flow_stmt.loc[name]
                break
        else:
            raise KeyError("Could not find capital expenditure field")
        
        fcf = ocf + capex  # typically capex is negative in statements
        return fcf
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nFull cash flow statement data:")
        print(cash_flow_stmt)
        raise

def forecast_fcf(historical_fcf, forecast_years, growth_rate=0.05):
    """
    Forecast future FCF using a constant growth rate from the last known FCF.
    """
    last_fcf = historical_fcf.iloc[0]  # most recent year
    forecasted_fcf = []
    for year in range(1, forecast_years+1):
        future_fcf = last_fcf * ((1 + growth_rate)**year)
        forecasted_fcf.append(future_fcf)
    return forecasted_fcf

def calculate_dcf(forecasted_fcf, wacc, terminal_growth=0.02):
    """
    Calculate the DCF using:
      1) Forecasted FCF
      2) Terminal Value based on perpetuity growth
      3) Discount each year's FCF
    Returns a dictionary containing the main outputs.
    """
    # 1. Present Value of forecasted FCF
    discount_factors = [(1 + wacc)**t for t in range(1, len(forecasted_fcf)+1)]
    pv_forecast = [fcf / df for fcf, df in zip(forecasted_fcf, discount_factors)]
    
    # 2. Terminal Value in the final year
    final_year_fcf = forecasted_fcf[-1]
    terminal_value = final_year_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / discount_factors[-1]
    
    # 3. Sum up all present values (Enterprise Value)
    enterprise_value = sum(pv_forecast) + pv_terminal
    
    # 4. For a full DCF, you'd subtract net debt to get equity value, 
    #    and then divide by shares outstanding. For illustration:
    #    net_debt = ...
    #    equity_value = enterprise_value - net_debt
    #    fair_value_per_share = equity_value / shares_outstanding
    
    return {
        "pv_forecast": pv_forecast,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value
    }

def forecast_metrics(historical_metrics, forecast_years=5, assumptions=None):
    """
    Forecast all metrics using growth assumptions
    """
    if assumptions is None:
        assumptions = {
            'revenue_growth': [0.05] * forecast_years,
            'gross_margin': [historical_metrics['Gross Margin %'].iloc[0]] * forecast_years,
            'ebit_margin': [historical_metrics['EBIT Margin %'].iloc[0]] * forecast_years,
            'da_percent': [0.05] * forecast_years,
            'fcf_margin': [historical_metrics['FCF Margin %'].iloc[0]] * forecast_years
        }
    
    # Get last year's values (in millions)
    last_revenue = historical_metrics['Revenue'].iloc[0]
    
    # Create empty DataFrame with all required columns
    years = [f"Year {i}" for i in range(1, forecast_years + 1)]
    forecast = pd.DataFrame(
        columns=[
            'Revenue', 'Gross Profit', 'EBIT', 'EBITDA', 'FCF',
            'Gross Margin %', 'EBIT Margin %', 'EBITDA Margin %', 'FCF Margin %'
        ],
        index=years
    )
    
    # Project metrics
    for year in range(forecast_years):
        # Revenue (in millions)
        revenue = last_revenue * (1 + assumptions['revenue_growth'][year])
        
        # Derive other metrics from revenue (all in millions)
        gross_profit = revenue * assumptions['gross_margin'][year]
        ebit = revenue * assumptions['ebit_margin'][year]
        da = revenue * assumptions['da_percent'][year]
        ebitda = ebit + da
        fcf = revenue * assumptions['fcf_margin'][year]
        
        # Update last_revenue for next iteration
        last_revenue = revenue
        
        # Store in forecast DataFrame
        forecast.loc[f"Year {year + 1}"] = [
            revenue,
            gross_profit,
            ebit,
            ebitda,
            fcf,
            assumptions['gross_margin'][year],
            assumptions['ebit_margin'][year],
            (ebitda / revenue),
            assumptions['fcf_margin'][year]
        ]
    
    return forecast, assumptions

def export_dcf_to_excel(output_dict, historical_metrics, forecast_metrics, assumptions, file_name="DCF_Model.xlsx"):
    """
    Exports the DCF calculation and metrics to Excel, including sensitivity analysis
    """
    with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
        workbook = writer.book
        num_format = workbook.add_format({'num_format': '#,##0.0'})
        percent_format = workbook.add_format({'num_format': '0.0%'})
        
        # Export historical metrics
        historical_metrics.to_excel(writer, sheet_name="Historical Metrics")
        worksheet = writer.sheets["Historical Metrics"]
        worksheet.set_column('B:F', 12, num_format)
        worksheet.set_column('G:J', 12, percent_format)
        
        # Export forecast metrics
        forecast_metrics.to_excel(writer, sheet_name="Forecast Metrics")
        worksheet = writer.sheets["Forecast Metrics"]
        worksheet.set_column('B:F', 12, num_format)
        worksheet.set_column('G:J', 12, percent_format)
        
        # Export DCF summary
        df_summary = pd.DataFrame({
            "Enterprise Value (MM)": [output_dict["enterprise_value"]],
            "Terminal Value (MM)": [output_dict["terminal_value"]],
            "PV of Terminal Value (MM)": [output_dict["pv_terminal"]]
        })
        df_summary.to_excel(writer, sheet_name="DCF Summary", index=False)
        worksheet = writer.sheets["DCF Summary"]
        worksheet.set_column('A:C', 20, num_format)
        
        # Create sensitivity analysis
        wacc_range = np.linspace(assumptions['wacc'] - 0.02, assumptions['wacc'] + 0.02, 5)
        growth_range = np.linspace(assumptions['terminal_growth'] - 0.01, assumptions['terminal_growth'] + 0.01, 5)
        
        sensitivity_data = []
        for w in wacc_range:
            row = []
            for g in growth_range:
                dcf_result = calculate_dcf(forecast_metrics['FCF'].values, w, g)
                row.append(dcf_result['enterprise_value'])
            sensitivity_data.append(row)
        
        df_sensitivity = pd.DataFrame(
            sensitivity_data,
            index=[f'WACC {w:.1%}' for w in wacc_range],
            columns=[f'Growth {g:.1%}' for g in growth_range]
        )
        
        # Export sensitivity analysis
        df_sensitivity.to_excel(writer, sheet_name="Sensitivity Analysis")
        worksheet = writer.sheets["Sensitivity Analysis"]
        worksheet.set_column('A:F', 15, num_format)
        
        # Export assumptions
        df_assumptions = pd.DataFrame({
            'Year': range(1, len(assumptions['revenue_growth']) + 1),
            'Revenue Growth': assumptions['revenue_growth'],
            'Gross Margin': assumptions['gross_margin'],
            'EBIT Margin': assumptions['ebit_margin'],
            'D&A %': assumptions['da_percent'],
            'FCF Margin': assumptions['fcf_margin']
        })
        df_assumptions.to_excel(writer, sheet_name="Assumptions", index=False)
        worksheet = writer.sheets["Assumptions"]
        worksheet.set_column('A:A', 8)
        worksheet.set_column('B:F', 15, percent_format)
    
    print(f"DCF model exported to {file_name} successfully!")

def main():
    # Get ticker from user
    ticker = get_ticker_input()
    
    # 1. Fetch financials
    income_stmt, cash_flow_stmt = fetch_financials(ticker)
    
    # Check if financial data is available
    if income_stmt is None or cash_flow_stmt is None:
        print("Exiting program due to lack of financial data.")
        return
    
    # 2. Calculate historical metrics
    historical_metrics = calculate_metrics(income_stmt, cash_flow_stmt)
    
    # Ask user for the number of forecast years
    forecast_years = int(input("Enter the number of forecast years: "))
    
    # 3. Set assumptions and forecast metrics
    assumptions = generate_assumptions_with_openai(historical_metrics, forecast_years)
    
    print("Generated Assumptions:")
    for key, value in assumptions.items():
        if isinstance(value, list):
            # Adjust the list to match the number of forecast years
            value = value[:forecast_years] + [0.05] * (forecast_years - len(value))
            print(f"\n{key}:")
            for year, val in enumerate(value, 1):
                new_value = input(f"Year {year} (current: {val:.2%}): ")
                if new_value:
                    value[year-1] = float(new_value) / 100
            assumptions[key] = value
        else:
            # Handle single values (wacc and terminal_growth)
            new_value = input(f"{key} (current: {value:.2%}): ")
            if new_value:
                assumptions[key] = float(new_value) / 100
    
    # Validate assumptions
    if assumptions['wacc'] <= assumptions['terminal_growth']:
        print("Error: WACC must be greater than terminal growth rate.")
        return
    
    # 4. Forecast metrics and calculate DCF
    forecast_data, _ = forecast_metrics(historical_metrics, forecast_years, assumptions)
    forecasted_fcf = forecast_data['FCF'].values
    dcf_results = calculate_dcf(forecasted_fcf, assumptions['wacc'], assumptions['terminal_growth'])
    
    # 5. Export to Excel
    export_dcf_to_excel(dcf_results, historical_metrics, forecast_data, assumptions, "DCF_Model.xlsx")

if __name__ == "__main__":
    main()