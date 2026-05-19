library(tidyverse)
library(grf) #for ATE (causal forest)

df <- read.csv('Step4_4_binarize_disease_column.csv')


#################ATE of education###################

#extracting significant features from the all-feature XGBoost set model
#mutate binary education variable
ate_df <- df %>% 
  dplyr::select(income_fam_2, bmi_2, frnd_sat_2, met_2,
                social_act_n_2, X25059.2.0, alcohol_2,
                social_act_2_pub, X25068.2.0, social_act_2_other,
                social_act_2_education, smoke_status_2, 
                social_act_2_religious, freq_visit_2,
                X25069.2.0, confide_2, X25077.2.0,
                age_2, X25103.2.0, X25095.2.0, X25066.2.0,
                X25093.2.0, X25067.2.0, X25061.2.0, emp_2, 
                X25102.2.0,
                ed_yr_2
  ) %>% 
  mutate(ed_b_2 = ifelse(ed_yr_2 >= 20, 1, 0)) 


#set up a function to compute z-statistics, p-values, and 95% confidence intervals
ate_sig <- function(ate_obj, alpha = 0.05) {
  est <- ate_obj["estimate"]
  se  <- ate_obj["std.err"]
  z <- est / se
  p <- 2 * (1 - pnorm(abs(z)))
  ci_lower <- est - qnorm(1 - alpha/2) * se
  ci_upper <- est + qnorm(1 - alpha/2) * se
  out <- data.frame(
    estimate = est,
    std.err  = se,
    z        = z,
    p.value  = p,
    ci.lower = ci_lower,
    ci.upper = ci_upper
  )
  
  return(out)
}

###########################################################
#######################friendship##########################
###########################################################

#covariates: standardize continuous variables
X_frnd_b <- ate_df %>% 
  select(-'frnd_sat_2', -'ed_b_2', -'ed_yr_2') %>%  
  mutate(across(
    where(is.numeric) & !where(~ all(.x %in% c(0,1))),
    scale
  ))

#outcome
Y_frnd_b <- ate_df$frnd_sat_2

#treatment: binarized education
W_frnd_b <- ate_df$ed_b_2

#fit a causal forest
set.seed(42)
CF_frnd_b <- causal_forest(X_frnd_b, 
                           Y_frnd_b,
                           W_frnd_b,
                           min.node.size = 100,
                           num.trees = 5000)


###########################################################
####################N of social act########################
###########################################################

#covariates: standardize continuous variables
X_Nact_b <- ate_df %>% 
  select(-'social_act_n_2', -'ed_b_2', -'ed_yr_2') %>%  
  mutate(across(
    where(is.numeric) & !where(~ all(.x %in% c(0,1))),
    scale
  ))

#outcome
Y_Nact_b <- ate_df$social_act_n_2

#treatment: binarized education
W_Nact_b <- ate_df$ed_b_2

#fit a causal forest
set.seed(42)
CF_Nact_b <- causal_forest(X_Nact_b, 
                           Y_Nact_b,
                           W_Nact_b,
                           min.node.size = 100,
                           num.trees = 5000)

###########################################################
##########################visit############################
###########################################################

#covariates: standardize continuous variables
X_vst_b <- ate_df %>% 
  select(-'freq_visit_2', -'ed_b_2', -'ed_yr_2') %>% 
  mutate(across(
    where(is.numeric) & !where(~ all(.x %in% c(0,1))),
    scale
  ))

#outcome
Y_vst_b <- ate_df$freq_visit_2

#treatment: binarized education
W_vst_b <- ate_df$ed_b_2

#fit a causal forest
set.seed(42)
CF_vst_b <- causal_forest(X_vst_b, 
                          Y_vst_b,
                          W_vst_b,
                          min.node.size = 100,
                          num.trees = 5000)

###########################################################
#########################confide###########################
###########################################################

#covariates: standardize continuous variables
X_cfd_b <- ate_df %>% 
  select(-'confide_2', -'ed_b_2', -'ed_yr_2') %>% 
  mutate(across(
    where(is.numeric) & !where(~ all(.x %in% c(0,1))),
    scale
  ))

#outcome
Y_cfd_b <- ate_df$confide_2

#treatment: binarized education
W_cfd_b <- ate_df$ed_b_2

#fit a causal forest
set.seed(42)
CF_cfd_b <- causal_forest(X_cfd_b, 
                          Y_cfd_b,
                          W_cfd_b,
                          min.node.size = 100, 
                          num.trees = 5000) 

###########################################################
####### Bonferroni correction across 4 ATE outcomes #######
###########################################################

ate_results <- rbind(
  Friendship_satisfaction = ate_sig(average_treatment_effect(CF_frnd_b)),
  N_social_activities     = ate_sig(average_treatment_effect(CF_Nact_b)),
  Frequency_of_visits     = ate_sig(average_treatment_effect(CF_vst_b)),
  Frequency_of_confiding  = ate_sig(average_treatment_effect(CF_cfd_b))
)

ate_results$p.bonf <- p.adjust(ate_results$p.value, method = "bonferroni")

ate_results$p.value <- formatC(ate_results$p.value, format = "e", digits = 3)
ate_results$p.bonf  <- formatC(ate_results$p.bonf,  format = "e", digits = 3)

print(ate_results)

