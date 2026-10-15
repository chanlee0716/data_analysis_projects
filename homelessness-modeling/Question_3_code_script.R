# Set up the plotting layout to have 2 rows and 3 columns
par(mfrow = c(2, 3))

# Generate component-plus-residual plots (ignoring interactions)
termplot(model, partial.resid = TRUE, se = TRUE, smooth = panel.smooth, ask = FALSE)

# --- add quadratic terms for interaction effects
model_nonlinear_inter <- lm(homelessness_rate_logit ~ econ_labor_unemp_rate_BLS +
                              dem_soc_ed_hsgrad_acs5yr_2017 +
                              econ_labor_medinc_acs5yr_2017 +
                              hou_mkt_medrent_acs5yr_2017 +
                              env_wea_avgtemp_summer_noaa +
                              econ_labor_unemp_rate_BLS:dem_soc_ed_hsgrad_acs5yr_2017 +
                              econ_labor_unemp_rate_BLS:econ_labor_medinc_acs5yr_2017 +
                              I(econ_labor_unemp_rate_BLS^2 * dem_soc_ed_hsgrad_acs5yr_2017) +
                              I(econ_labor_unemp_rate_BLS * dem_soc_ed_hsgrad_acs5yr_2017^2),
                            data = data_2017)
anova(model, model_nonlinear_inter)

# --- Step 2: Add Quadratic Terms for Main Effects ---
model_quad_main <- lm(homelessness_rate_logit ~ 
                        econ_labor_unemp_rate_BLS + I(econ_labor_unemp_rate_BLS^2) +
                        dem_soc_ed_hsgrad_acs5yr_2017 + I(dem_soc_ed_hsgrad_acs5yr_2017^2) +
                        econ_labor_medinc_acs5yr_2017 + I(econ_labor_medinc_acs5yr_2017^2) +
                        hou_mkt_medrent_acs5yr_2017 + I(hou_mkt_medrent_acs5yr_2017^2) +
                        env_wea_avgtemp_summer_noaa + I(env_wea_avgtemp_summer_noaa^2) +
                        econ_labor_unemp_rate_BLS:dem_soc_ed_hsgrad_acs5yr_2017 +
                        econ_labor_unemp_rate_BLS:econ_labor_medinc_acs5yr_2017,
                      data = data_2017)
anova(model, model_quad_main)  # Test quadratic main effects

# --- Step 3: Add Quadratic Terms for Interaction Effects ---
model_quad_inter <- lm(homelessness_rate_logit ~ 
                         econ_labor_unemp_rate_BLS + I(econ_labor_unemp_rate_BLS^2) +
                         dem_soc_ed_hsgrad_acs5yr_2017 + I(dem_soc_ed_hsgrad_acs5yr_2017^2) +
                         econ_labor_medinc_acs5yr_2017 + I(econ_labor_medinc_acs5yr_2017^2) +
                         hou_mkt_medrent_acs5yr_2017 + I(hou_mkt_medrent_acs5yr_2017^2) +
                         env_wea_avgtemp_summer_noaa + I(env_wea_avgtemp_summer_noaa^2) +
                         econ_labor_unemp_rate_BLS:dem_soc_ed_hsgrad_acs5yr_2017 +
                         I(econ_labor_unemp_rate_BLS^2 * dem_soc_ed_hsgrad_acs5yr_2017) +
                         econ_labor_unemp_rate_BLS:econ_labor_medinc_acs5yr_2017 +
                         I(econ_labor_unemp_rate_BLS * econ_labor_medinc_acs5yr_2017^2),
                       data = data_2017)
anova(model, model_quad_inter) # Test quadratic interaction effects

# --- Step 4: Add Cubic Terms for Main Effects ---
model_cubic_main <- lm(homelessness_rate_logit ~ 
                         econ_labor_unemp_rate_BLS + I(econ_labor_unemp_rate_BLS^2) + I(econ_labor_unemp_rate_BLS^3) +
                         dem_soc_ed_hsgrad_acs5yr_2017 + I(dem_soc_ed_hsgrad_acs5yr_2017^2) + I(dem_soc_ed_hsgrad_acs5yr_2017^3) +
                         econ_labor_medinc_acs5yr_2017 + I(econ_labor_medinc_acs5yr_2017^2) + I(econ_labor_medinc_acs5yr_2017^3) +
                         hou_mkt_medrent_acs5yr_2017 + I(hou_mkt_medrent_acs5yr_2017^2) + I(hou_mkt_medrent_acs5yr_2017^3) +
                         env_wea_avgtemp_summer_noaa + I(env_wea_avgtemp_summer_noaa^2) + I(env_wea_avgtemp_summer_noaa^3) +
                         econ_labor_unemp_rate_BLS:dem_soc_ed_hsgrad_acs5yr_2017 +
                         econ_labor_unemp_rate_BLS:econ_labor_medinc_acs5yr_2017,
                       data = data_2017)
anova(model_quad_main, model_cubic_main)  # Test cubic main effects



# --- Step 5: Test cubic interaction effects
model_cubic_inter <- lm(homelessness_rate_logit ~ 
                          econ_labor_unemp_rate_BLS + I(econ_labor_unemp_rate_BLS^2) + I(econ_labor_unemp_rate_BLS^3) +
                          dem_soc_ed_hsgrad_acs5yr_2017 + I(dem_soc_ed_hsgrad_acs5yr_2017^2) + I(dem_soc_ed_hsgrad_acs5yr_2017^3) +
                          econ_labor_medinc_acs5yr_2017 + I(econ_labor_medinc_acs5yr_2017^2) + I(econ_labor_medinc_acs5yr_2017^3) +
                          hou_mkt_medrent_acs5yr_2017 + I(hou_mkt_medrent_acs5yr_2017^2) + I(hou_mkt_medrent_acs5yr_2017^3) +
                          env_wea_avgtemp_summer_noaa + I(env_wea_avgtemp_summer_noaa^2) + I(env_wea_avgtemp_summer_noaa^3) +
                          econ_labor_unemp_rate_BLS:dem_soc_ed_hsgrad_acs5yr_2017 +
                          I(econ_labor_unemp_rate_BLS^2 * dem_soc_ed_hsgrad_acs5yr_2017) +
                          econ_labor_unemp_rate_BLS:econ_labor_medinc_acs5yr_2017 +
                          I(econ_labor_unemp_rate_BLS * econ_labor_medinc_acs5yr_2017^2),
                        data = data_2017)
anova(model_quad_inter, model_cubic_inter)  