data_2017_2 <- data_2017
#data_2017_2$state_abr <- as.factor(data_2017_2$state_abr)

# obtain all possible predictor variables
non_predictors <- c(
  "homelessness_rate", "homelessness_rate_logit", 
  "homelessness_rate_log", "homelessness_rate_adj", 
  "homelessness_rate_density", "year", "ID", "cocnumber", "panelvar", "state_abr",
  "pit_hless_pit_hud_share", "pit_tot_hless_pit_hud", "pit_hless_balance",
  "missing", "pit_shelt_balance", "pit_unshelt_balance", "pit_shelt_pit_hud_share",
  "pit_unshelt_pit_hud_share", "flag_d_hless", "flag_xt_hless", "flag_d_shelt",
  "flag_xt_shelt", "flag_d_unshelt", "flag_xt_unshelt", "pit_ind_chronic_hless_pit_hud",
  "pit_perfam_chronic_hless_pit_hud", "dem_pop_pop_census", "pit_tot_hless_pit_hud",
  "pit_ind_hless_pit_hud",
  "pit_perfam_shelt_pit_hud",
  "pit_perfam_unshelt_pit_hud",
  "pit_perfam_hless_pit_hud",
  "coctag",
  "hou_pol_perm_bed_hic_hud",
  "hou_pol_temp_bed_hic_hud",
  "dem_pop_female_census",
  "dem_pop_senior_census",
  "dem_soc_hispanic_census",
  "dem_soc_ed_somecoll_acs5yr",
  "econ_labor_unemp_pop_BLS",
  "pit_hless_pit_hud_share",
  "missing",
  "dem_pop_male_census_share",
  "dem_soc_white_census_share",
  "dem_soc_other_census_share",
  "dem_health_mhlth_chr_share_2017",
  "dem_health_excesdrink_chr_2017",
  "dem_health_ins_acs5yr_2017",
  "dem_soc_ed_bach_acs5yr_2017",
  "dem_soc_ed_somecoll_acs5yr_2017",
  "dem_soc_ed_hsgrad_acs5yr_2017",
  "dem_soc_ed_lesshs_acs5yr_2017",
  "dem_soc_singparent_acs5yr_2017",
  "dem_soc_singadult_acs5yr_2017",
  "dem_soc_vet_acs5yr_2017",
  "econ_labor_incineq_acs5yr_2017",
  "econ_labor_topskill_acs5yr_2017",
  "econ_labor_midskill_acs5yr_2017",
  "econ_labor_unskill_acs5yr_2017",
  "econ_labor_medinc_acs5yr_2017",
  "econ_sn_cashasst_acs5yr_2017",
  "hou_mkt_rentshare_acs5yr_2017",
  "hou_mkt_rentvacancy_acs5yr_2017",
  "hou_mkt_ovrcrowd_acs5yr_2017",
  "hou_mkt_homeage_acs5yr_2017",
  "hou_pol_hlessconduct_beg",
  "cpi_2017",
  "odd_flag",
  "unbalance_flag",
  "econ_labor_unskill_acs5yr_diff",
  "econ_sn_cashasst_acs5yr_diff",
  "hou_mkt_rentvacancy_acs5yr_diff",
  "hou_mkt_homeage_acs5yr_diff",
  "hou_mkt_homeage1940_acs5yr_2017",
  "hou_mkt_homeval_acs5yr_2017",
  "hou_mkt_medrent_acs5yr_2017",
  "hou_mkt_utility_acs5yr_2017",
  "hou_mkt_burden_own_acs5yr_2017",
  "hou_mkt_burden_own_acs5yr_diff",
  "hou_mkt_burden_sev_own_acs5yr_2017",
  "hou_mkt_burden_sev_own_acs5yr_diff",
  "hou_mkt_burden_rent_acs5yr_2017",
  "hou_mkt_burden_sev_rent_acs5yr_2017",
  "hou_mkt_burden_sev_rent_acs5yr_diff",
  "fedfundcoc_flag",
  "hou_mkt_homeval_xt",
  "hou_mkt_burden_own_xt",
  "hou_mkt_burden_sev_own_xt",
  "hou_mkt_burden_rent_xt",
  "hou_mkt_burden_sev_rent_xt",
  "hou_mkt_homeage_xt",
  "hou_mkt_homeage1940_xt",
  "hou_mkt_medrent_xt",
  "hou_mkt_utility_xt",
  "hou_mkt_rentshare_xt",
  "hou_mkt_rentvacancy_xt",
  "hou_mkt_ovrcrowd_xt",
  "econ_labor_midskill_xt",
  "econ_labor_incineq_xt",
  "hou_mkt_burden_sev_own_acs_2017",
  "hou_mkt_burden_sev_own_acs_diff",
  "hou_mkt_burden_sev_rent_acs_2017",
  "hou_mkt_burden_sev_rent_acs_diff",
  "econ_sn_cashasst_xt",
  "dem_soc_ed_bach_xt",
  "dem_soc_ed_somecoll_xt",
  "dem_soc_ed_hsgrad_xt",
  "dem_soc_ed_lesshs_xt",
  "dem_health_ins_xt",
  "dem_soc_singadult_xt",
  "dem_soc_singparent_xt",
  "dem_soc_vet_xt",
  "d_hou_mkt_burden_sev_own_xt",
  "d_hou_mkt_rentvacancy_xt",
  "d_hou_mkt_ovrcrowd_xt",
  "d_econ_sn_cashasst_xt",
  "d_dem_soc_vet_xt",
  "dem_soc_ed_lessbach_xt",
  "tight_high_cost_rental_mkt",
  "suburban",
  "rural"
)
predictor_vars <- setdiff(names(data_2017_2), non_predictors)

missing_threshold <- 0.5  # Set threshold (e.g., remove variables with >50% missing)
predictor_vars <- predictor_vars[colMeans(is.na(data_2017_2[predictor_vars])) < missing_threshold]

######### LASSO ######### 
set.seed(123)

# Combine X and y into a single dataframe
lasso_data <- data_2017_2[, c("homelessness_rate_logit", predictor_vars)]

# Remove rows with any missing values
lasso_data <- na.omit(lasso_data)

# Redefine X and y after removing missing rows
y <- lasso_data$homelessness_rate_logit
X <- model.matrix(~ ., data = lasso_data)[, -1]  # Remove intercept column

# Keep only rows where predictors are not missing
valid_rows <- complete.cases(data_2017_2[predictor_vars])

# Subset y and X accordingly
y <- data_2017_2$homelessness_rate_logit[valid_rows]
X <- model.matrix(~ ., data = data_2017_2[valid_rows, predictor_vars])[, -1]

set.seed(123)
cv_lasso <- cv.glmnet(X, y, alpha = 1)
plot(cv_lasso)

best_lambda <- cv_lasso$lambda.min
lasso_model <- glmnet(X, y, alpha = 1, lambda = best_lambda)
selected_vars <- rownames(coef(lasso_model))[coef(lasso_model)[, 1] != 0]

cat("Selected Predictors:\n")
print(setdiff(selected_vars, "(Intercept)"))

### Model
reduced_model_lasso <- lm(homelessness_rate_logit ~ 
                            econ_urb_urbanicity + 
                            hou_pol_bed_th_hic_hud +
                            pit_miss +
                            dem_pop_child_census_share +
                            dem_soc_racetwo_census_share +
                            dem_health_alcdeath_IMHE_2015 +
                            econ_sn_cashasst_acs5yr_2012 +
                            hou_mkt_rentshare_acs5yr_2012 +
                            hou_mkt_burden_sev_own_acs_2012 +
                            hou_pol_hudunit_psh_hud_share +
                            d_fhfa_hpi_2009 +
                            d_dem_pop_adult_census_share +
                            d_dem_pop_female_census_share +
                            d_env_wea_precip_noaa + 
                            sub_west_coast +
                            sub_west_census,
                          data = data_2017_2)
#summary(reduced_model_lasso)

# add interactions
reduced_model_lasso_1 <- lm(homelessness_rate_logit ~ 
                              econ_urb_urbanicity + 
                              pit_miss +
                              dem_pop_child_census_share +
                              dem_soc_racetwo_census_share +
                              dem_health_alcdeath_IMHE_2015 +
                              hou_mkt_burden_sev_own_acs_2012 +
                              hou_pol_hudunit_psh_hud_share +
                              d_fhfa_hpi_2009 +
                              d_dem_pop_female_census_share +
                              sub_west_coast +
                              econ_urb_urbanicity:hou_mkt_burden_sev_own_acs_2012 + 
                              d_dem_pop_adult_census_share:sub_west_coast   ,                      
                            data = data_2017_2)
summary(reduced_model_lasso_1)


### Compare with Original Model
print(paste("Adjusted R-squared for Original Model:", round(summary(model)$adj.r.squared, 3)))
print(paste("Adjusted R-squared for Model using Lasso:", round(summary(reduced_model_lasso_1)$adj.r.squared, 3)))

print(paste("RSE for Original Model:", round(summary(model)$sigma, 3)))
print(paste("RSE for Model using Lasso:", round(summary(reduced_model_lasso_1)$sigma, 3)))

# 5. Cross-validation (10-Fold CV)
train_control <- trainControl(method = "cv", number = 10)
cv_model1 <- train(homelessness_rate_logit ~ econ_labor_unemp_rate_BLS + 
                     dem_soc_ed_hsgrad_acs5yr_2017 +
                     econ_labor_medinc_acs5yr_2017 +
                     hou_mkt_medrent_acs5yr_2017 +
                     env_wea_avgtemp_summer_noaa +
                     econ_labor_unemp_rate_BLS:econ_labor_medinc_acs5yr_2017 +
                     econ_labor_unemp_rate_BLS:dem_soc_ed_hsgrad_acs5yr_2017, data = data_2017_2, method = "lm", trControl = train_control)

cv_model3 <- train(homelessness_rate_logit ~ econ_urb_urbanicity + 
                     pit_miss +
                     dem_pop_child_census_share +
                     dem_soc_racetwo_census_share +
                     dem_health_alcdeath_IMHE_2015 +
                     hou_mkt_burden_sev_own_acs_2012 +
                     hou_pol_hudunit_psh_hud_share +
                     d_fhfa_hpi_2009 +
                     d_dem_pop_female_census_share +
                     sub_west_coast +
                     econ_urb_urbanicity:hou_mkt_burden_sev_own_acs_2012 + 
                     d_dem_pop_adult_census_share:sub_west_coast, data = data_2017_2, method = "lm", trControl = train_control)


print("CV Results for Original Model:")
print(cv_model1$results$RMSE)
print("CV Results for Model using Lasso:")
print(cv_model3$results$RMSE)