library(tidyverse)
library(pROC) #for ROC
library(car) #for multicollinearity

df <- read.csv('Step4_4_binarize_disease_column.csv')

names(df) <- make.names(names(df))

#binarize variables 
df <- df %>% mutate(ethnicity_b_2 = ifelse(ethnicity_0 >= 2, 1, 0),
                    emp_b_2 = ifelse(emp_2 >= 2, 0, 1),
                    ed_b_2 = ifelse(ed_yr_2 >= 20, 1, 0),
                    income_fam_2 = ifelse(income_fam_2 >= 1, income_fam_2, NA)              
)

####################logit######################

#logistic regression on fluid intelligence to validate feature importance results from XGBoost

#fit a logistic regression model
#continuous variables were standardized
df_logit <- glm(fluid_2_p10 ~ 
                  scale(ed_yr_2) + scale(met_2) + scale(age_2) +
                  scale(alcohol_2) + ethnicity_b_2 +
                  scale(income_fam_2) + scale(X25092.2.0) + gender +
                  scale(X25059.2.0) + scale(X25068.2.0) +
                  scale(X25094.2.0) + emp_b_2 +
                  scale(X25095.2.0) + scale(frnd_sat_2) + 
                  scale(X25070.2.0) + scale(X25098.2.0) +
                  scale(X25069.2.0) + scale(bmi_2),
                family = binomial,
                data = df)

#see results
summary(df_logit)

raw_p <- summary(df_logit)$coefficients[, "Pr(>|z|)"]

#calculating the odds ratio and 95% confidence interval, and rounding them to three decimal places
df_logit_res <- as.data.frame(round(
  cbind(
    OR  = exp(coef(df_logit)),
    LCI = exp(confint(df_logit)[, 1]),
    UCI = exp(confint(df_logit)[, 2])
  ), 3))

df_logit_res$p_value <- formatC(raw_p, format = "e", digits = 3)
df_logit_res$p_bonf  <- formatC(p.adjust(raw_p, method = "bonferroni"), 
                                format = "e", digits = 3)

print(df_logit_res)

#check multicollinearity
vif(df_logit)

#calculate AUC
pred_prob <- predict(df_logit, type = "response")
y_true <- df$fluid_2_p10
roc_obj <- roc(y_true, pred_prob)
auc(roc_obj)



#logistic regression on fluid intelligence to examine the association with social network features

#select social network features and significant features in classifying fluid intelligence as covariates
df_sn <- df %>% 
  dplyr::select(frnd_sat_2, social_act_n_2,
                freq_visit_2, confide_2, ed_yr_2, met_2, age_2, alcohol_2,
                ethnicity_b_2, income_fam_2, 
                X25092.2.0, gender,
                X25059.2.0, X25068.2.0, X25094.2.0,
                emp_b_2, X25095.2.0, 
                X25070.2.0, X25098.2.0, X25069.2.0,
                bmi_2,
                fluid_2_p10)

#continuous variables were standardized
df_sn <- df_sn  %>%  
  mutate(across(
    where(is.numeric) & !where(~ all(.x %in% c(0,1))),
    scale
  ))

#fit a logistic regression model
df_glm  <- glm(fluid_2_p10 ~ .,
               family = binomial,
               data = df_sn)

#see results
summary(df_glm)

# Calculating the odds ratio and 95% confidence interval, and rounding them to three decimal places
raw_p_glm <- summary(df_glm)$coefficients[, "Pr(>|z|)"]

df_glm_res <- as.data.frame(round(
  cbind(
    OR  = exp(coef(df_glm)),
    LCI = exp(confint(df_glm)[, 1]),
    UCI = exp(confint(df_glm)[, 2])
  ), 3))

df_glm_res$p_value <- formatC(raw_p_glm, format = "e", digits = 3)
df_glm_res$p_bonf  <- formatC(p.adjust(raw_p_glm, method = "bonferroni"), 
                              format = "e", digits = 3)

print(df_glm_res)

#check multicollinearity
vif(df_glm)

#calculate AUC
pred_df_glm  <- predict(df_glm, type = "response")
y_true <- df$fluid_2_p10
roc_obj_ed <- roc(y_true, pred_df_glm)
auc(roc_obj_ed)

