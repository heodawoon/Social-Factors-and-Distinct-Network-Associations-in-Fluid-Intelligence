library(tidyverse)
library(mediation) #for causal mediation analysis


df <- read.csv('Step4_4_binarize_disease_column.csv')
names(df) <- make.names(names(df))


#################Mediation of education###################

#confounders: SHAP-selected variables from the Gf model,
#excluding educational attainment and the four mediators
covariates <- c("met_2", "age_2", "alcohol_2", "ethnicity_b_2", "income_fam_2",
                "X25092.2.0", "gender", "X25059.2.0", "X25068.2.0", "X25094.2.0",
                "emp_b_2", "X25095.2.0", "X25070.2.0", "X25098.2.0", "X25069.2.0",
                "bmi_2")

#recoding identical to the logistic regression and ATE scripts
#complete cases across all variables, so the four models share one sample
med_df <- df %>%
  mutate(ed_b_2        = ifelse(ed_yr_2 >= 20, 1, 0),
         ethnicity_b_2 = ifelse(ethnicity_0 >= 2, 1, 0),
         emp_b_2       = ifelse(emp_2 >= 2, 0, 1),
         income_fam_2  = ifelse(income_fam_2 >= 1, income_fam_2, NA)) %>%
  dplyr::select(fluid_2_p10, ed_b_2,
                frnd_sat_2, social_act_n_2, freq_visit_2, confide_2,
                all_of(covariates)) %>%
  tidyr::drop_na()

cat("Sample size used:", nrow(med_df), "\n") #expected: 5492

cov_terms <- paste(covariates, collapse = " + ")


#set up a function to extract the average causal mediation effect,
#average direct effect, total effect, and proportion mediated
med_sig <- function(med_obj) {
  out <- data.frame(
    ACME          = med_obj$d.avg,
    ACME.lower    = med_obj$d.avg.ci[1],
    ACME.upper    = med_obj$d.avg.ci[2],
    ACME.p        = med_obj$d.avg.p,
    ADE           = med_obj$z.avg,
    ADE.lower     = med_obj$z.avg.ci[1],
    ADE.upper     = med_obj$z.avg.ci[2],
    total.effect  = med_obj$tau.coef,
    total.lower   = med_obj$tau.ci[1],
    total.upper   = med_obj$tau.ci[2],
    prop.mediated = med_obj$n.avg,
    prop.lower    = med_obj$n.avg.ci[1],
    prop.upper    = med_obj$n.avg.ci[2],
    prop.p        = med_obj$n.avg.p
  )
  
  return(out)
}

###########################################################
#######################friendship##########################
###########################################################

#mediator model: mediator ~ treatment + confounders
M_frnd_b <- lm(as.formula(paste("frnd_sat_2 ~ ed_b_2 +", cov_terms)),
               data = med_df)

#outcome model: Gf decile group ~ treatment + mediator + confounders
O_frnd_b <- glm(as.formula(paste("fluid_2_p10 ~ ed_b_2 + frnd_sat_2 +", cov_terms)),
                family = binomial,
                data = med_df)

#estimate the mediation effect with percentile bootstrap CIs
set.seed(42)
MD_frnd_b <- mediate(M_frnd_b,
                     O_frnd_b,
                     treat        = "ed_b_2",
                     mediator     = "frnd_sat_2",
                     boot         = TRUE,
                     boot.ci.type = "perc",
                     sims         = 1000)

###########################################################
####################N of social act########################
###########################################################

#mediator model: mediator ~ treatment + confounders
M_Nact_b <- lm(as.formula(paste("social_act_n_2 ~ ed_b_2 +", cov_terms)),
               data = med_df)

#outcome model: Gf decile group ~ treatment + mediator + confounders
O_Nact_b <- glm(as.formula(paste("fluid_2_p10 ~ ed_b_2 + social_act_n_2 +", cov_terms)),
                family = binomial,
                data = med_df)

#estimate the mediation effect with percentile bootstrap CIs
set.seed(42)
MD_Nact_b <- mediate(M_Nact_b,
                     O_Nact_b,
                     treat        = "ed_b_2",
                     mediator     = "social_act_n_2",
                     boot         = TRUE,
                     boot.ci.type = "perc",
                     sims         = 1000)

###########################################################
##########################visit############################
###########################################################

#mediator model: mediator ~ treatment + confounders
M_vst_b <- lm(as.formula(paste("freq_visit_2 ~ ed_b_2 +", cov_terms)),
              data = med_df)

#outcome model: Gf decile group ~ treatment + mediator + confounders
O_vst_b <- glm(as.formula(paste("fluid_2_p10 ~ ed_b_2 + freq_visit_2 +", cov_terms)),
               family = binomial,
               data = med_df)

#estimate the mediation effect with percentile bootstrap CIs
set.seed(42)
MD_vst_b <- mediate(M_vst_b,
                    O_vst_b,
                    treat        = "ed_b_2",
                    mediator     = "freq_visit_2",
                    boot         = TRUE,
                    boot.ci.type = "perc",
                    sims         = 1000)

###########################################################
#########################confide###########################
###########################################################

#mediator model: mediator ~ treatment + confounders
M_cfd_b <- lm(as.formula(paste("confide_2 ~ ed_b_2 +", cov_terms)),
              data = med_df)

#outcome model: Gf decile group ~ treatment + mediator + confounders
O_cfd_b <- glm(as.formula(paste("fluid_2_p10 ~ ed_b_2 + confide_2 +", cov_terms)),
               family = binomial,
               data = med_df)

#estimate the mediation effect with percentile bootstrap CIs
set.seed(42)
MD_cfd_b <- mediate(M_cfd_b,
                    O_cfd_b,
                    treat        = "ed_b_2",
                    mediator     = "confide_2",
                    boot         = TRUE,
                    boot.ci.type = "perc",
                    sims         = 1000)

###########################################################
############ summary across the four mediators ############
###########################################################

#full output for each mediator
summary(MD_frnd_b)
summary(MD_Nact_b)
summary(MD_vst_b)
summary(MD_cfd_b)

#because each mediator is modeled separately,
#the proportions mediated are not additive
#the smallest p-value attainable with 1,000 resamples is 0.002
med_results <- rbind(
  Friendship_satisfaction = med_sig(MD_frnd_b),
  N_social_activities     = med_sig(MD_Nact_b),
  Frequency_of_visits     = med_sig(MD_vst_b),
  Frequency_of_confiding  = med_sig(MD_cfd_b)
)

print(med_results)

