# ============================================================
# Descriptive_table.R
#
# Generates participant characteristics tables
# (Supplementary Tables 10-13 in the manuscript).
#
# Two grouping definitions are supported:
#   (1) Gf group:       top 10% vs. bottom 10% of fluid intelligence
#   (2) Education group: 4-year college graduates (>=20 years) vs. non-graduates
#
# For each grouping, the script produces a LaTeX table with:
#   - continuous variables: mean +/- SD
#   - categorical variables: N (%)
#   - test statistic with degrees of freedom
#       * Welch's two-sample t-test for continuous variables
#       * Pearson's chi-squared test for categorical variables
#   - p-value
#
# Usage:
#   Rscript Descriptive_table.R <input_csv> <output_dir>
#
# Example:
#   Rscript Descriptive_table.R data/Step4_4_binarize_disease_column.csv output/
#
# If no arguments are provided, the script falls back to the default
# paths defined below (edit as needed for local use).
# ============================================================

# -------- 0) Packages --------
# install.packages(c("readr", "dplyr", "gtsummary", "kableExtra"))
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(gtsummary)
  library(kableExtra)
})

# -------- 1) Command-line arguments --------
args <- commandArgs(trailingOnly = TRUE)

input_csv  <- if (length(args) >= 1) args[1] else "data/Step4_4_binarize_disease_column.csv"
output_dir <- if (length(args) >= 2) args[2] else "output/"

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# -------- 2) Variable definitions --------
categorical_vars <- c(
  "gender", "ethnicity_0", "marital_2", "emp_2", "income_fam_2", "lone_2",
  "social_act_2_sport", "social_act_2_pub", "social_act_2_religious",
  "social_act_2_education", "social_act_2_other", "smoke_status_2",
  "glass_lenses_2", "eye_issue_2", "hearing_issue_2", "hearing_issue_bg_2",
  "hearing_aid_2"
)

continuous_vars <- c(
  "age_2", "ed_yr_2", "fncl_sat_2", "hthcare_2", "social_act_n_2", "freq_visit_2",
  "confide_2", "fam_sat_2", "frnd_sat_2", "N_fam_2", "alcohol_2", "phq2_2",
  "hlth_sat_2", "sleep_2", "bmi_2", "met_2", "fluid_2"
)

var_order <- c(
  "fluid_2", "age_2", "gender", "ethnicity_0", "marital_2", "ed_yr_2", "emp_2",
  "income_fam_2", "hthcare_2", "fncl_sat_2", "N_fam_2", "social_act_n_2",
  "social_act_2_sport", "social_act_2_pub", "social_act_2_religious",
  "social_act_2_education", "social_act_2_other", "freq_visit_2", "fam_sat_2",
  "frnd_sat_2", "lone_2", "confide_2", "phq2_2", "hlth_sat_2", "met_2",
  "alcohol_2", "smoke_status_2", "sleep_2", "bmi_2", "eye_issue_2",
  "glass_lenses_2", "hearing_issue_2", "hearing_issue_bg_2", "hearing_aid_2"
)

# -------- 3) Load data --------
message("Reading: ", input_csv)
df <- read_csv(input_csv, show_col_types = FALSE)

# -------- 4) Helper: test statistic + df --------
get_test_stat <- function(data, variable, by, ...) {
  x <- data[[variable]]
  g <- data[[by]]
  
  ok <- !is.na(x) & !is.na(g)
  x <- x[ok]; g <- g[ok]
  
  if (is.numeric(x)) {
    # Welch's two-sample t-test
    tt <- t.test(x ~ g)
    sprintf("t(%.1f) = %.2f",
            as.numeric(tt$parameter), as.numeric(tt$statistic))
  } else {
    # Pearson's chi-squared test
    tab <- table(x, g)
    ct  <- suppressWarnings(chisq.test(tab))
    sprintf("$\\chi^2$(%d) = %.2f",
            as.integer(ct$parameter), as.numeric(ct$statistic))
  }
}

# -------- 5) Generic table-builder --------
build_descriptive_table <- function(data, group_var,
                                    top_label    = "Top 10%",
                                    bottom_label = "Bottom 10%") {
  
  # ensure factor levels
  data <- data %>%
    mutate(
      !!group_var := factor(.data[[group_var]], levels = c(top_label, bottom_label))
    ) %>%
    filter(!is.na(.data[[group_var]])) %>%
    mutate(across(all_of(categorical_vars), as.factor))
  
  n_top    <- sum(data[[group_var]] == top_label)
  n_bottom <- sum(data[[group_var]] == bottom_label)
  n_all    <- nrow(data)
  
  tab <- data %>%
    select(all_of(c(group_var, var_order))) %>%
    tbl_summary(
      by = all_of(group_var),
      type = list(
        all_of(continuous_vars)  ~ "continuous",
        all_of(categorical_vars) ~ "categorical"
      ),
      statistic = list(
        all_continuous()  ~ "{mean} \u00b1 {sd}",
        all_categorical() ~ "{n} ({p}%)"
      ),
      digits = list(
        all_continuous()  ~ 2,
        all_categorical() ~ c(0, 2)
      ),
      missing = "no"
    ) %>%
    add_overall(last = FALSE, col_label = "All") %>%
    add_stat(
      fns      = everything() ~ get_test_stat,
      location = everything() ~ "label"
    ) %>%
    add_p(
      test = list(
        all_continuous()  ~ "t.test",
        all_categorical() ~ "chisq.test"
      ),
      pvalue_fun = ~ style_pvalue(.x, digits = 3)
    ) %>%
    modify_header(
      label      ~ "Variable",
      stat_0     ~ paste0("All (N = ",                 format(n_all,    big.mark = ","), ")"),
      stat_1     ~ paste0(top_label,    " (N = ",      format(n_top,    big.mark = ","), ")"),
      stat_2     ~ paste0(bottom_label, " (N = ",      format(n_bottom, big.mark = ","), ")"),
      add_stat_1 ~ "Test statistic (df)",
      p.value    ~ "p-value"
    )
  
  tab
}

# -------- 6) Export helper --------
export_latex <- function(tbl, out_path) {
  latex_tbl <- tbl %>%
    as_kable_extra(format = "latex", booktabs = TRUE, escape = FALSE) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"),
                  font_size = 9)
  cat(latex_tbl, file = out_path)
  message("Saved: ", out_path)
  invisible(latex_tbl)
}

# -------- 7) Required-column check --------
needed <- c("fluid_2_p10", "ed_yr_2", var_order)
miss <- setdiff(needed, names(df))
if (length(miss) > 0) {
  stop("Missing columns in input data: ", paste(miss, collapse = ", "))
}

# ============================================================
# Table A: Gf group (top 10% vs bottom 10%)
# ============================================================
df_gf <- df %>%
  mutate(
    gf_group = case_when(
      fluid_2_p10 == 1 ~ "Top 10%",
      fluid_2_p10 == 0 ~ "Bottom 10%",
      TRUE             ~ NA_character_
    )
  )

tab_gf <- build_descriptive_table(df_gf, group_var = "gf_group")
export_latex(tab_gf, file.path(output_dir, "descriptive_table_fluid2_top_bottom10.tex"))

# ============================================================
# Table B: Education group (>=20 years vs <20 years)
# ============================================================
df_ed <- df %>%
  mutate(
    ed20_group = case_when(
      is.na(ed_yr_2) ~ NA_character_,
      ed_yr_2 >= 20  ~ "Top 10%",
      ed_yr_2 <  20  ~ "Bottom 10%"
    )
  )

tab_ed <- build_descriptive_table(df_ed, group_var = "ed20_group")
export_latex(tab_ed, file.path(output_dir, "descriptive_table_edyr20.tex"))

message("Done.")
