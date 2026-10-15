data_2017 <- data_2017[-1880, ]

model <- lm(homelessness_rate_logit ~ 
              econ_labor_unemp_rate_BLS + 
              dem_soc_ed_hsgrad_acs5yr_2017 +
              econ_labor_medinc_acs5yr_2017 +
              hou_mkt_medrent_acs5yr_2017 +
              env_wea_avgtemp_summer_noaa +
              econ_labor_unemp_rate_BLS:econ_labor_medinc_acs5yr_2017 +
              econ_labor_unemp_rate_BLS:dem_soc_ed_hsgrad_acs5yr_2017, 
            data = data_2017)

summary(model)

## Diagnostics

# Set up the plotting layout: 2 rows, 1st row with 2 plots, 2nd row with 1 plot
layout(matrix(c(1, 2), 2, 1, byrow = TRUE), heights = c(1, 2))  # First row gets half the height, second row gets double

# First row: Check homoscedasticity (Residuals vs Fitted) and Normality of Residuals (QQ plot)
par(mfrow = c(1, 2))  # Set this row to 1 row, 2 columns for side-by-side plots

# Residuals vs Fitted
plot(fitted(model), resid(model),
     main = "Residuals vs Fitted",
     xlab = "Fitted Values",
     ylab = "Residuals")
abline(h = 0, col = "red")

# QQ plot for normality of residuals
qqnorm(resid(model))
qqline(resid(model), col = "red")

# Reset the layout for the second row
par(mfrow = c(1, 1))  # Set back to 1 plot per row

# Second row: Diagnostic plots for outliers
# Outlier Test
outlierTest(model)

# Influence Index Plot
influenceIndexPlot(model)


