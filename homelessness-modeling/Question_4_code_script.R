# Fit a model that interacts rural with the main predictors.
# (Note: Our base model 'model' is linear with two interactions already.)
model_rural_inter <- lm(homelessness_rate_logit ~ 
                          rural * (econ_labor_unemp_rate_BLS +
                                     dem_soc_ed_hsgrad_acs5yr_2017 +
                                     econ_labor_medinc_acs5yr_2017 +
                                     hou_mkt_medrent_acs5yr_2017 +
                                     env_wea_avgtemp_summer_noaa) +
                          econ_labor_unemp_rate_BLS:dem_soc_ed_hsgrad_acs5yr_2017 +
                          econ_labor_unemp_rate_BLS:econ_labor_medinc_acs5yr_2017,
                        data = data_2017)
summary(model_rural_inter)