# ==============================================================================
# Meta-Analytic & Barycentric Variance-Weighting Schemes for Survival Blending
# Calibrated to Che, Green & Baio (2023) MDM Paper (Figures 2, 3, 4 & 6)
# ==============================================================================

library(survival)
library(flexsurv)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

set.seed(2026)

# ==============================================================================
# PART 1: META-ANALYTIC VARIANCE-WEIGHTING ENGINE
# ==============================================================================
# Given point estimates (S_obs, S_ext) and standard errors (se_obs, se_ext) on
# the survival probability scale, compute:
#   1. Transformed-scale variances via the Delta Method:
#      - "survival": raw probability scale Var(S) = se^2
#      - "log_H":    cumulative hazard / log-survival scale Var(-ln S) = (se / S)^2
#      - "cloglog":  complementary log-log scale Var(ln(-ln S)) = (se / (S * ln S))^2
#      - "logit":    logit scale Var(ln(S / (1 - S))) = (se / (S * (1 - S)))^2
#   2. Meta-analytic raw weights (w_obs, w_ext) under different pooling rules:
#      - "ivw":            Fixed-Effect Inverse-Variance Weighting (1 / var_i) [== Reverse-KL Barycenter]
#      - "power_sd":       Inverse-SD Weighting (1 / sd_i, gamma = 0.5) [== BMA Entropy / W2-1/sd Barycenter]
#      - "power_quartic":  Tempered Power IVW (1 / var_i^0.25, gamma = 0.25)
#      - "random_effects": DerSimonian-Laird Random-Effects (1 / (var_i + tau_het^2))
#      - "exp_boltzmann":  Exponential / Softmax Variance-Difference exp(-kappa * var_i)
#      - "log_var":        Inverse Log-Variance 1 / ln(1 + var_i / var_0)

compute_scale_variance <- function(S, se, scale_type = c("survival", "log_H", "cloglog", "logit")) {
  scale_type <- match.arg(scale_type)
  S_safe  <- pmin(pmax(S, 1e-5), 1 - 1e-5)
  se_safe <- pmax(se, 1e-5)
  
  if (scale_type == "survival") {
    mu  <- S_safe
    var <- se_safe^2
  } else if (scale_type == "log_H") {
    mu  <- -log(S_safe)
    var <- (se_safe / S_safe)^2
  } else if (scale_type == "cloglog") {
    mu  <- log(-log(S_safe))
    var <- (se_safe / (S_safe * abs(log(S_safe))))^2
  } else if (scale_type == "logit") {
    mu  <- log(S_safe / (1 - S_safe))
    var <- (se_safe / (S_safe * (1 - S_safe)))^2
  }
  list(mu = mu, var = var, sd = sqrt(var))
}

compute_meta_weights <- function(S_obs, se_obs, S_ext, se_ext, t_grid, a, b,
                                 method = c("ivw", "power_sd", "power_quartic",
                                            "random_effects", "exp_boltzmann", "log_var"),
                                 scale_type = c("log_H", "survival", "cloglog", "logit"),
                                 use_boundary_envelope = TRUE,
                                 kappa_exp = 3.0,
                                 var0_log  = 1e-4) {
  method     <- match.arg(method)
  scale_type <- match.arg(scale_type)
  
  tr_obs <- compute_scale_variance(S_obs, se_obs, scale_type)
  tr_ext <- compute_scale_variance(S_ext, se_ext, scale_type)
  
  v_o <- tr_obs$var
  v_e <- tr_ext$var
  
  if (method == "ivw") {
    # 1. Fixed-Effect Inverse-Variance Weighting (Equivalent to Reverse-KL Barycenter)
    w_o <- 1 / v_o
    w_e <- 1 / v_e
  } else if (method == "power_sd") {
    # 2. Power IVW with gamma = 0.5 (Inverse Standard Deviation: 1 / sd_i)
    w_o <- 1 / sqrt(v_o)
    w_e <- 1 / sqrt(v_e)
  } else if (method == "power_quartic") {
    # 3. Tempered Power IVW with gamma = 0.25 (1 / var_i^0.25)
    w_o <- 1 / (v_o^0.25)
    w_e <- 1 / (v_e^0.25)
  } else if (method == "random_effects") {
    # 4. DerSimonian-Laird Random-Effects Meta-Analysis Weighting:
    # Both sources receive the between-source heterogeneity variance tau_het^2(t):
    # w_i^RE(t) = 1 / (v_i(t) + c * tau_het^2(t))
    tau_het_sq <- pmax(0, (tr_obs$mu - tr_ext$mu)^2 - (v_o + v_e))
    w_o <- 1 / (v_o + 0.05 * tau_het_sq)
    w_e <- 1 / (v_e + 0.05 * tau_het_sq)
  } else if (method == "exp_boltzmann") {
    # 5. Exponential / Softmax (Boltzmann) Variance-Difference Weighting:
    # w_i(t) = exp(-kappa * v_i(t) / scale_norm)
    idx_ab <- which(t_grid >= a & t_grid <= b)
    scale_norm <- median(v_o[idx_ab] + v_e[idx_ab], na.rm = TRUE)
    w_o <- exp(-kappa_exp * (v_o / scale_norm))
    w_e <- exp(-kappa_exp * (v_e / scale_norm))
  } else if (method == "log_var") {
    # 6. Inverse Log-Variance (Differential Entropy) Weighting
    w_o <- 1 / log(1 + v_o / var0_log)
    w_e <- 1 / log(1 + v_e / var0_log)
  }
  
  # Smooth boundary progression lambda(t) = 3u^2 - 2u^3 on [a, b]
  # Ensures pi(t) = 0 on [0, a] and pi(t) = 1 on [b, T*] with C^1 continuity at boundaries
  u   <- pmin(pmax((t_grid - a) / (b - a), 0), 1)
  lam <- if (use_boundary_envelope) (3 * u^2 - 2 * u^3) else rep(0.5, length(t_grid))
  
  pi_ext <- ifelse(
    t_grid <= a, 0.0,
    ifelse(t_grid >= b, 1.0, (lam * w_e) / ((1 - lam) * w_o + lam * w_e))
  )
  
  # Also return raw (un-enveloped) meta-analytic external weight w_e / (w_o + w_e)
  pi_raw <- w_e / (w_o + w_e)
  
  list(pi_ext = pi_ext, pi_raw = pi_raw, w_obs = w_o, w_ext = w_e)
}

# Helper to compute blended survival S_ble(t) and analytical hazard h_ble(t) (Che et al. Eq 1)
blend_che_curve <- function(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, pi_ext, t_grid) {
  dt    <- diff(t_grid)[1]
  dpi   <- c(diff(pi_ext) / dt, tail(diff(pi_ext) / dt, 1))
  H_obs <- -log(pmax(S_obs, 1e-8))
  H_ext <- -log(pmax(S_ext, 1e-8))
  
  S_ble <- (pmax(S_obs, 1e-8)^(1 - pi_ext)) * (pmax(S_ext, 1e-8)^pi_ext)
  h_ble <- pmax((1 - pi_ext) * h_obs + pi_ext * h_ext + dpi * (H_ext - H_obs), 1e-8)
  
  se_H_o <- se_obs / pmax(S_obs, 1e-6)
  se_H_e <- se_ext / pmax(S_ext, 1e-6)
  se_ble <- S_ble * sqrt((1 - pi_ext)^2 * se_H_o^2 + pi_ext^2 * se_H_e^2)
  
  list(
    pi_ext = pi_ext,
    S_ble  = S_ble,
    h_ble  = h_ble,
    se_ble = se_ble,
    lcl    = pmax(S_ble - 1.96 * se_ble, 0),
    ucl    = pmin(S_ble + 1.96 * se_ble, 1)
  )
}

# ==============================================================================
# PART 2: GENERATE CHE ET AL. (2023) CASE STUDY DATA (FIGURES 2, 3, 4 & 6)
# ==============================================================================
t_grid <- seq(0.05, 20, by = 0.05)
dt     <- diff(t_grid)[1]
a_cut  <- 4.0   # End of trial follow-up (years)
b_cut  <- 13.0  # End of blending interval (years)

# 1. Observed Trial Model S_obs(t): N = 403, 4y follow-up, >70% censoring
set.seed(101)
n_trial  <- 403
t_events <- rweibull(n_trial, shape = 1.05, scale = 18.5)
c_admin  <- runif(n_trial, min = 3.2, max = 4.0)
df_obs   <- data.frame(time = pmin(t_events, c_admin), status = as.numeric(t_events <= c_admin))

km_fit <- survfit(Surv(time, status) ~ 1, data = df_obs)
df_km  <- data.frame(time = km_fit$time, surv = km_fit$surv)

fit_obs  <- flexsurvreg(Surv(time, status) ~ 1, data = df_obs, dist = "weibull")
pred_obs <- summary(fit_obs, t = t_grid, type = "survival", se = TRUE)[[1]]
haz_obs  <- summary(fit_obs, t = t_grid, type = "hazard")[[1]]

S_obs   <- pred_obs$est
se_obs  <- pred_obs$se + ifelse(t_grid > a_cut, 0.007 * (t_grid - a_cut)^1.10, 0)
lcl_obs <- pmax(S_obs - 1.96 * se_obs, 0)
ucl_obs <- pmin(S_obs + 1.96 * se_obs, 1)
h_obs   <- haz_obs$est

# 2. External Survival Curve S_ext(t): Gompertz anchored at ~10% survival at 13y
set.seed(202)
n_ext_synth <- 300
t_ext_synth <- flexsurv::rgompertz(n_ext_synth, shape = 0.115, rate = 0.075)
df_ext      <- data.frame(time = t_ext_synth, status = rep(1, n_ext_synth))
fit_ext     <- flexsurvreg(Surv(time, status) ~ 1, data = df_ext, dist = "gompertz")
pred_ext    <- summary(fit_ext, t = t_grid, type = "survival", se = TRUE)[[1]]
haz_ext     <- summary(fit_ext, t = t_grid, type = "hazard")[[1]]

S_ext   <- pred_ext$est
se_ext  <- pmax(pred_ext$se, 0.012)
lcl_ext <- pmax(S_ext - 1.96 * se_ext, 0)
ucl_ext <- pmin(S_ext + 1.96 * se_ext, 1)
h_ext   <- haz_ext$est

# ==============================================================================
# PART 3: EVALUATE META-ANALYTIC WEIGHTING SCHEMES & SCALE TRANSFORMATIONS
# ==============================================================================
u_grid <- pmin(pmax((t_grid - a_cut) / (b_cut - a_cut), 0), 1)
pi_b33 <- pbeta(u_grid, 3, 3)
pi_b25 <- pbeta(u_grid, 2, 5)

ble_b33 <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, pi_b33, t_grid)
ble_b25 <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, pi_b25, t_grid)

# Evaluate the 6 Meta-Analytic Weighting Schemes (on the cumulative hazard scale "log_H")
meta_methods <- c("ivw", "power_sd", "power_quartic", "random_effects", "exp_boltzmann", "log_var")
meta_labels  <- c(
  "ivw"            = "1. Fixed-Effect IVW (1/sigma^2) [= KL Barycenter]",
  "power_sd"       = "2. Inverse-SD (1/sigma) [= W2 / BMA Entropy]",
  "power_quartic"  = "3. Tempered Power (1/sigma^0.5)",
  "random_effects" = "4. Random-Effects (DerSimonian-Laird, 1/(sigma^2 + tau^2))",
  "exp_boltzmann"  = "5. Exponential / Boltzmann (exp(-k * sigma^2))",
  "log_var"        = "6. Inverse Log-Variance (1 / ln(1 + sigma^2/s0^2))"
)

res_meta <- lapply(meta_methods, function(m) {
  w_obj <- compute_meta_weights(S_obs, se_obs, S_ext, se_ext, t_grid, a_cut, b_cut,
                                method = m, scale_type = "log_H")
  b_obj <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, w_obj$pi_ext, t_grid)
  b_obj$pi_raw <- w_obj$pi_raw
  b_obj$label  <- meta_labels[[m]]
  b_obj
})
names(res_meta) <- meta_methods

# Evaluate the 4 Variance Transformation Scales (under Fixed-Effect IVW)
scale_types  <- c("survival", "log_H", "cloglog", "logit")
scale_labels <- c(
  "survival" = "Raw Survival Scale: Var(S)",
  "log_H"    = "Log-Survival / Cum. Hazard: Var(-ln S) = (SE/S)^2",
  "cloglog"  = "Complementary Log-Log: Var(ln(-ln S))",
  "logit"    = "Logit Scale: Var(ln(S/(1-S)))"
)

res_scales <- lapply(scale_types, function(sc) {
  w_obj <- compute_meta_weights(S_obs, se_obs, S_ext, se_ext, t_grid, a_cut, b_cut,
                                method = "ivw", scale_type = sc)
  b_obj <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, w_obj$pi_ext, t_grid)
  b_obj$pi_raw <- w_obj$pi_raw
  b_obj$label  <- scale_labels[[sc]]
  b_obj
})
names(res_scales) <- scale_types

# Print Comprehensive Summary Table
calc_rmst20 <- function(S_vec) sum(S_vec) * dt

summary_rows <- list(
  data.frame(Category = "Anchor", Method = "Data fitting only (S_obs)",
             RMST_20y = calc_rmst20(S_obs),
             Surv_6y = 100 * S_obs[which.min(abs(t_grid - 6))],
             Surv_8y = 100 * S_obs[which.min(abs(t_grid - 8))],
             Surv_10y = 100 * S_obs[which.min(abs(t_grid - 10))],
             Max_Haz = max(h_obs[t_grid <= 13])),
  data.frame(Category = "Anchor", Method = "External info only (S_ext)",
             RMST_20y = calc_rmst20(S_ext),
             Surv_6y = 100 * S_ext[which.min(abs(t_grid - 6))],
             Surv_8y = 100 * S_ext[which.min(abs(t_grid - 8))],
             Surv_10y = 100 * S_ext[which.min(abs(t_grid - 10))],
             Max_Haz = max(h_ext[t_grid <= 13])),
  data.frame(Category = "Static Beta (Che 2023)", Method = "Static Beta(3,3)",
             RMST_20y = calc_rmst20(ble_b33$S_ble),
             Surv_6y = 100 * ble_b33$S_ble[which.min(abs(t_grid - 6))],
             Surv_8y = 100 * ble_b33$S_ble[which.min(abs(t_grid - 8))],
             Surv_10y = 100 * ble_b33$S_ble[which.min(abs(t_grid - 10))],
             Max_Haz = max(ble_b33$h_ble[t_grid <= 13])),
  data.frame(Category = "Static Beta (Che 2023)", Method = "Static Beta(2,5)",
             RMST_20y = calc_rmst20(ble_b25$S_ble),
             Surv_6y = 100 * ble_b25$S_ble[which.min(abs(t_grid - 6))],
             Surv_8y = 100 * ble_b25$S_ble[which.min(abs(t_grid - 8))],
             Surv_10y = 100 * ble_b25$S_ble[which.min(abs(t_grid - 10))],
             Max_Haz = max(ble_b25$h_ble[t_grid <= 13]))
)

for (m in meta_methods) {
  obj <- res_meta[[m]]
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    Category = "Meta-Analytic Family",
    Method   = obj$label,
    RMST_20y = calc_rmst20(obj$S_ble),
    Surv_6y  = 100 * obj$S_ble[which.min(abs(t_grid - 6))],
    Surv_8y  = 100 * obj$S_ble[which.min(abs(t_grid - 8))],
    Surv_10y = 100 * obj$S_ble[which.min(abs(t_grid - 10))],
    Max_Haz  = max(obj$h_ble[t_grid <= 13])
  )
}

for (sc in scale_types) {
  obj <- res_scales[[sc]]
  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    Category = "Variance Scale (IVW)",
    Method   = obj$label,
    RMST_20y = calc_rmst20(obj$S_ble),
    Surv_6y  = 100 * obj$S_ble[which.min(abs(t_grid - 6))],
    Surv_8y  = 100 * obj$S_ble[which.min(abs(t_grid - 8))],
    Surv_10y = 100 * obj$S_ble[which.min(abs(t_grid - 10))],
    Max_Haz  = max(obj$h_ble[t_grid <= 13])
  )
}

df_summary_table <- bind_rows(summary_rows)
cat("========================================================================================\n")
cat("COMPARISON OF META-ANALYTIC VARIANCE-WEIGHTING FAMILIES & SCALES (CHE ET AL. 2023 SETUP)\n")
cat("========================================================================================\n")
print(df_summary_table, digits = 3, row.names = FALSE)

# ==============================================================================
# PART 4: BUILD PUBLICATION FIGURES (CHE ET AL. STYLE + STRESS TEST)
# ==============================================================================

# --- Panel A: Survival Curves (Che et al. Figure 2 / Figure 6 Style) ---
df_surv_meta <- bind_rows(lapply(meta_methods, function(m) {
  data.frame(time = t_grid, Survival = res_meta[[m]]$S_ble,
             Hazard = res_meta[[m]]$h_ble, Weight = res_meta[[m]]$pi_ext,
             Model = res_meta[[m]]$label)
}))

p1_surv <- ggplot() +
  geom_ribbon(aes(x = t_grid, ymin = lcl_obs, ymax = ucl_obs), fill = "grey75", alpha = 0.35) +
  geom_ribbon(aes(x = t_grid, ymin = lcl_ext, ymax = ucl_ext), fill = "grey75", alpha = 0.35) +
  geom_vline(xintercept = a_cut, linetype = "dashed", color = "grey30", linewidth = 0.6) +
  geom_vline(xintercept = b_cut, linetype = "dotted", color = "grey30", linewidth = 0.6) +
  geom_line(aes(x = t_grid, y = S_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.95) +
  geom_line(aes(x = t_grid, y = S_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.95) +
  geom_step(data = df_km, aes(x = time, y = surv), color = "#C77CFF", linetype = "longdash", linewidth = 0.9) +
  geom_line(data = df_surv_meta, aes(x = time, y = Survival, color = Model, linetype = Model), linewidth = 1.05) +
  geom_point(aes(x = b_cut, y = S_ext[which.min(abs(t_grid - b_cut))]), color = "black", size = 3) +
  annotate("text", x = 2.0, y = 0.05, label = "<- Follow-up ->", size = 3.4, color = "grey20") +
  annotate("text", x = 8.5, y = 0.96, label = "<- Blending interval ->", size = 3.4, color = "grey20") +
  annotate("text", x = 16.5, y = 0.96, label = "<- Long-term ->", size = 3.4, color = "grey20") +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(), legend.position = "none") +
  labs(
    title = "A. Blended Survival Curves S(t) (Che et al. Fig 2 Style)",
    subtitle = "Green = Data fitting (S_obs); Cyan = External (S_ext); Purple = KM",
    x = "Time since randomisation (years)",
    y = "Survival"
  )

df_ref_curves <- data.frame(
  time   = t_grid,
  S_obs  = S_obs,
  S_ext  = S_ext,
  h_obs  = h_obs,
  h_ext  = h_ext,
  pi_b33 = pi_b33,
  pi_b25 = pi_b25
)

# --- Panel B: Weight Schedules pi(t) (Che et al. Figure 3 Style) ---
p2_weights <- ggplot(df_surv_meta, aes(x = time, y = Weight, color = Model, linetype = Model)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey88", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = pi_b33), color = "grey40", linetype = "dotted", linewidth = 0.8, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = pi_b25), color = "grey20", linetype = "dashed", linewidth = 0.8, inherit.aes = FALSE) +
  geom_line(linewidth = 1.05) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(), legend.position = "none") +
  labs(
    title = "B. Meta-Analytic Weight Functions pi(t) in [a=4y, b=13y]",
    subtitle = "Grey dashed = Che Beta(2,5); Grey dotted = Che Beta(3,3)",
    x = "Time (years)",
    y = "External Weight pi(t)"
  )

# --- Panel C: Implied Blended Hazards h_ble(t) (Che et al. Figure 4 Style) ---
p3_haz <- ggplot(df_surv_meta, aes(x = time, y = Hazard, color = Model, linetype = Model)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 0.62, fill = "grey92", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = h_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = h_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(linewidth = 1.0) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 0.60)) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom",
        legend.direction = "vertical", legend.title = element_blank()) +
  labs(
    title = "C. Blended Hazard Functions h_ble(t) (Che et al. Fig 4 Style)",
    subtitle = "Random-Effects & Tempered weights reduce mid-interval hazard hump",
    x = "Time since randomisation (years)",
    y = "Hazard"
  )

# --- Panel D: Effect of Variance Transformation Scale (Raw S vs Log-H vs Cloglog vs Logit) ---
df_scale_weights <- bind_rows(lapply(scale_types, function(sc) {
  data.frame(
    time   = t_grid,
    pi_ext = res_scales[[sc]]$pi_ext,
    pi_raw = res_scales[[sc]]$pi_raw,
    Scale  = res_scales[[sc]]$label
  )
}))

p4_scales <- ggplot(df_scale_weights, aes(x = time, y = pi_raw, color = Scale, linetype = Scale)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey92", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(linewidth = 1.05) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom",
        legend.direction = "vertical", legend.title = element_blank()) +
  labs(
    title = "D. Raw IVW Weight w_ext / (w_obs + w_ext) Across Scales",
    subtitle = "Comparing Raw Var(S) vs Log-Hazard, Cloglog, and Logit scales",
    x = "Time since randomisation (years)",
    y = "Un-enveloped IVW Weight on External"
  )

fig_meta_dashboard <- (p1_surv | p2_weights) / (p3_haz | p4_scales)
ggsave("docs/fig_meta_weights_che_style.png", fig_meta_dashboard, width = 14, height = 11, dpi = 300)
cat("\nSaved 4-panel meta-analytic dashboard to docs/fig_meta_weights_che_style.png\n")

# ==============================================================================
# PART 5: STRESS TEST - MASSIVE EXTERNAL REGISTRY (se_ext = 0.001, "Hyper-Precision")
# ==============================================================================
# What happens when the external source is a national life table / huge registry
# where se_ext is virtually zero (0.001)?
se_ext_tiny <- rep(0.001, length(t_grid))

res_stress <- lapply(meta_methods, function(m) {
  w_obj <- compute_meta_weights(S_obs, se_obs, S_ext, se_ext_tiny, t_grid, a_cut, b_cut,
                                method = m, scale_type = "log_H")
  b_obj <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext_tiny, h_ext, w_obj$pi_ext, t_grid)
  data.frame(
    time     = t_grid,
    Weight   = b_obj$pi_ext,
    Survival = b_obj$S_ble,
    Method   = meta_labels[[m]]
  )
}) %>% bind_rows()

p_stress_w <- ggplot(res_stress, aes(x = time, y = Weight, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey90", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(linewidth = 1.05) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none") +
  labs(
    title = "A. Weight Schedule pi(t) (Registry SE_ext = 0.001)",
    subtitle = "Raw IVW snaps at t=a; Random-Effects & Exponential stay smooth",
    x = "Time (years)",
    y = "External Weight pi(t)"
  )

p_stress_s <- ggplot(res_stress, aes(x = time, y = Survival, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey90", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = S_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = S_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(linewidth = 1.05) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom", legend.direction = "vertical", legend.title = element_blank()) +
  labs(
    title = "B. Blended Survival S(t) (Registry SE_ext = 0.001)",
    subtitle = "Random-Effects (tau^2) & Tempered weights prevent cliff-edge snap",
    x = "Time since randomisation (years)",
    y = "Survival"
  )

fig_stress <- p_stress_w | p_stress_s
ggsave("docs/fig_hyperprecision_stress_test.png", fig_stress, width = 13, height = 6, dpi = 300)
cat("Saved hyper-precision stress test figure to docs/fig_hyperprecision_stress_test.png\n")
