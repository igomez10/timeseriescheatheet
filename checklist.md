TIME SERIES EXAM CHEAT SHEET

1. THE "DATA SANITY" CHECK (First 5 Minutes)
Before modeling, verify the basics so your models don't fail silently.
- Check the Frequency: Is the data daily, weekly, or monthly? If statsmodels throws an error, you probably forgot to set .asfreq('D').
- Hunt for Hidden NaNs: Missing data breaks ARIMA and Holt-Winters. Use the .ffill() method to bridge gaps.
- Identify the Horizon: How many steps forward does the .csv require? Set your 'steps' variable once and leave it alone.

2. STATIONARITY & DIFFERENCING (d term)
Auto-ARIMA usually figures this out, but you need to know if it's making the right call based on the Dickey-Fuller (ADF) test.
- p-value <= 0.05: The data is stationary. No differencing needed (d = 0).
- p-value > 0.05: The data has a trend or non-stationary variance. Differencing is required (d >= 1).

3. DECODING ACF AND PACF PLOTS (p and q terms)
Use these plots on your stationary (differenced) data to verify the p (Auto-Regressive) and q (Moving Average) terms.
- Finding the AR term (p): Look at the PACF plot.
	- If the PACF plot has a sharp drop-off after a few lags, but the ACF tapers down gradually, you have an AR process. The cutoff lag is your p value.
- Finding the MA term (q): Look at the ACF plot.
	- If the ACF plot has a sharp drop-off after a few lags, but the PACF tapers down gradually, you have an MA process. The cutoff lag is your q value.
- Spotting Seasonality (m term): Look at the ACF plot.
	- Significant spikes repeating at regular intervals (e.g., every 7th lag) indicate strong seasonality.

4. EVALUATING MODEL OUTPUTS & RESIDUALS
Don't blindly trust the lowest RMSE.
- The "Flatline" Warning: If your ARIMA/Prophet forecast is a perfectly straight horizontal line, it failed to capture the signal. Switch to Holt-Winters or adjust seasonality parameters.
- Residual Diagnostics: Plot the residuals (Actuals minus Predictions). They should look like pure, random white noise. If they look like a sine wave, you missed the seasonality.

5. THE SUBMISSION CHECKLIST (Don't lose easy points)
Passing the auto-grader is half the battle.
- Notebook Execution: Does the notebook run top-to-bottom without a single error? Restart the kernel and run all cells before submitting.
- Format Match: Does your submission.csv have the exact same number of rows as the test file?
- Column Names: Are the index and headers named exactly what the professor asked for?