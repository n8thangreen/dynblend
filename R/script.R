# ==============================================================================
# Dynamic Survival Blending: Approach 1 vs. Approach 2
# ==============================================================================

# Required libraries
# install.packages(c("survival", "flexsurv", "ggplot2", "dplyr", "tidyr", "patchwork"))
library(survival)
library(flexsurv)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

# ------------------------------------------------------------------------------
# 1. Simulate Observed Clinical Trial and External Registry Data
# ------------------------------------------------------------------------------
set.seed(42)

# Observed trial cohort (median follow-up ~15m, heavy censoring after 24m)
n_obs    <- 350
true_obs <- rweibull(n_obs, shape = 0.9, scale = 42)
cens_obs <- runif(n_obs, min = 12, max = 24)
t_obs    <- pmin(true_obs, cens_obs)
d_obs    <- as.numeric(true_obs <= cens_obs)
df_obs   <- data.frame(time = t_obs, status = d_obs)

# Mature external cohort (long follow-up up to 84m, stable variance)
n_ext    <- 1500
true_ext <- rweibull(n_ext, shape = 1.25, scale = 22)
cens_ext <- runif(n_ext, min = 60, max = 84)
t_ext    <- pmin(true_ext, cens_ext)
d_ext    <- as.numeric(true_ext <= cens_ext)
df_ext   <- data.frame(time = t_ext, status = d_ext)

# ------------------------------------------------------------------------------
# 2. Fit Survival Models & Extract Survival + Standard Errors
# ------------------------------------------------------------------------------
fit_obs <- flexsurvreg(Surv(time, status) ~ 1, data = df_obs, dist = "weibull")
fit_ext <- flexsurvreg(Surv(time, status) ~ 1, data = df_ext, dist = "weibull")

# Define evaluation timeline (0.1 to 84 months = 7-year horizon)
t_grid <- seq(0.1, 84, by = 0.2)
dt     <- diff(t_grid)[1]

# Extract estimates and standard errors
pred_obs <- summary(fit_obs, t = t_grid, type = "survival", se = TRUE)[[1]]
pred_ext <- summary(fit_ext, t = t_grid, type = "survival", se = TRUE)[[1]]

S_obs  <- pred_obs$est
se_obs <- pmax(pred_obs$se, 1e-4) # Trial uncertainty expands past 24m

S_ext  <- pred_ext$est
se_ext <- pmax(pred_ext$se, 1e-4) # Registry uncertainty remains tight

# Define blending window
t1 <- 18  # Blending starts at 18 months
t2 <- 54  # Blending completes at 54 months

# ------------------------------------------------------------------------------
# 3. APPROACH 1: KL Divergence as the "Speedometer" (Paces Linear Blending)
# ------------------------------------------------------------------------------
# Pointwise Gaussian KL divergence between (S_obs, se_obs^2) and (S_ext, se_ext^2)
kl_pointwise <- 0.5 * (
  log((se_ext^2) / (se_obs^2)) +
    ((se_obs^2) + (S_obs - S_ext)^2) / (se_ext^2) - 1
)
kl_pointwise <- pmax(kl_pointwise, 0)

idx_blend <- which(t_grid >= t1 & t_grid <= t2)

# Apply a smooth windowing factor so derivative vanishes at boundaries t1 and t2
# This guarantees C^1 differentiability (no kinks or corners at 54m)
taper <- sin(pi * (t_grid[idx_blend] - t1) / (t2 - t1))^2
cum_kl_tapered <- cumsum(kl_pointwise[idx_blend] * taper) * dt

pi_app1 <- rep(0, length(t_grid))
pi_app1[t_grid > t2]   <- 1
pi_app1[idx_blend]     <- cum_kl_tapered / max(cum_kl_tapered)

# Compute Approach 1 blended survival curve (Arithmetic mixture)
S_blend_app1 <- (1 - pi_app1) * S_obs + pi_app1 * S_ext

# ------------------------------------------------------------------------------
# 4. APPROACH 2: Information Barycenter (Precision-Weighted Intermediate Dist)
# ------------------------------------------------------------------------------
# Uses a smooth schedule parameter lambda(t) in [0, 1] across [t1, t2]
lambda_t <- rep(0, length(t_grid))
lambda_t[t_grid > t2] <- 1
# Smoothstep (Hermite) transition parameter:
u <- pmin(pmax((t_grid[idx_blend] - t1) / (t2 - t1), 0), 1)
lambda_t[idx_blend] <- 3 * u^2 - 2 * u^3

# Convert standard errors to precisions (inverse variances)
prec_obs <- 1 / (se_obs^2)
prec_ext <- 1 / (se_ext^2)

# Solve for the intermediate Gaussian distribution M_t that minimizes:
# (1 - lambda)*D_KL(M || Obs) + lambda*D_KL(M || Ext)
# Precision of M_t:
prec_M <- (1 - lambda_t) * prec_obs + lambda_t * prec_ext
var_M  <- 1 / prec_M

# Mean of M_t (The precision-weighted blended survival curve):
S_blend_app2 <- var_M * (
  (1 - lambda_t) * prec_obs * S_obs + 
    lambda_t       * prec_ext * S_ext
)

# Benchmark: Paper's static Beta(5,5) over [t1, t2] for comparison
pi_beta_static <- pbeta(pmin(pmax((t_grid - t1) / (t2 - t1), 0), 1), shape1 = 5, shape2 = 5)
S_blend_beta   <- (1 - pi_beta_static) * S_obs + pi_beta_static * S_ext

# ------------------------------------------------------------------------------
# 5. Implied Hazard Functions & RMST Summary
# ------------------------------------------------------------------------------
# Calculate implied hazard h(t) = -S'(t) / S(t)
calc_implied_hazard <- function(S_vec, dt_step = dt) {
  dS <- -c(diff(S_vec) / dt_step, tail(diff(S_vec) / dt_step, 1))
  pmax(dS / pmax(S_vec, 1e-6), 0)
}

h_obs  <- calc_implied_hazard(S_obs)
h_ext  <- calc_implied_hazard(S_ext)
h_app1 <- calc_implied_hazard(S_blend_app1)
h_app2 <- calc_implied_hazard(S_blend_app2)
h_beta <- calc_implied_hazard(S_blend_beta)

# 7-Year Restricted Mean Survival Time (RMST in Years)
calc_rmst_years <- function(S_vec, dt_step = dt) {
  sum(S_vec) * dt_step / 12
}

rmst_comparison <- data.frame(
  Method = c("Internal Extrapolated (Unadjusted)", 
             "External Baseline Anchor", 
             "Static Beta(5,5) [Zhu / Che benchmark]", 
             "Approach 1 (KL Paced / Tapered Clock)", 
             "Approach 2 (KL Precision Barycenter)"),
  `7-Year RMST (Years)` = c(
    calc_rmst_years(S_obs),
    calc_rmst_years(S_ext),
    calc_rmst_years(S_blend_beta),
    calc_rmst_years(S_blend_app1),
    calc_rmst_years(S_blend_app2)
  )
)

cat("\n====================================================================\n")
cat("7-Year RMST Comparison across Blending Frameworks\n")
cat("====================================================================\n")
print(rmst_comparison, row.names = FALSE)

# ------------------------------------------------------------------------------
# 6. Multi-Panel Comparative Visualization
# ------------------------------------------------------------------------------
plot_df <- data.frame(
  time     = rep(t_grid, 5),
  survival = c(S_obs, S_ext, S_blend_beta, S_blend_app1, S_blend_app2),
  hazard   = c(h_obs, h_ext, h_beta, h_app1, h_app2),
  model    = rep(c("Extrapolated Observed", "External Anchor", 
                   "Static Beta(5,5)", "Approach 1 (KL Paced)", 
                   "Approach 2 (KL Barycenter)"), each = length(t_grid))
)

custom_colors <- c(
  "Extrapolated Observed"     = "firebrick",
  "External Anchor"           = "navy",
  "Static Beta(5,5)"          = "purple",
  "Approach 1 (KL Paced)"      = "forestgreen",
  "Approach 2 (KL Barycenter)" = "darkorange"
)

custom_lines <- c(
  "Extrapolated Observed"     = "dashed",
  "External Anchor"           = "dashed",
  "Static Beta(5,5)"          = "dotted",
  "Approach 1 (KL Paced)"      = "solid",
  "Approach 2 (KL Barycenter)" = "solid"
)

# Panel A: Transition Weight Schedules
weights_df <- data.frame(
  time       = t_grid,
  Beta_5_5   = pi_beta_static,
  App1_KL    = pi_app1,
  App2_Sched = lambda_t
) %>%
  filter(time >= t1 - 4 & time <= t2 + 4) %>%
  pivot_longer(-time, names_to = "Weight_Type", values_to = "Weight")

p1 <- ggplot(weights_df, aes(x = time, y = Weight, color = Weight_Type)) +
  geom_vline(xintercept = c(t1, t2), linetype = "dotted", color = "grey40") +
  geom_line(linewidth = 1) +
  scale_color_manual(
    values = c("Beta_5_5" = "purple", "App1_KL" = "forestgreen", "App2_Sched" = "darkorange"),
    labels = c("Static Beta(5,5)", "App 1: KL-Paced π(t)", "App 2: Barycenter λ(t)")
  ) +
  theme_minimal(base_size = 12) +
  labs(title = "A. Weight Schedules across Blending Window",
       x = "Time (Months)", y = "External Weight") +
  theme(legend.position = "bottom", legend.title = element_blank())

# Panel B: Survival Curves S(t)
p2 <- ggplot(plot_df, aes(x = time, y = survival, color = model, linetype = model)) +
  geom_rect(aes(xmin = t1, xmax = t2, ymin = 0, ymax = 1),
            fill = "grey90", alpha = 0.015, color = NA, inherit.aes = FALSE) +
  geom_vline(xintercept = c(t1, t2), linetype = "dotted", color = "grey40") +
  geom_line(linewidth = 1) +
  scale_color_manual(values = custom_colors) +
  scale_linetype_manual(values = custom_lines) +
  theme_minimal(base_size = 12) +
  labs(title = "B. Blended Survival Extrapolations S(t)",
       subtitle = paste0("Shaded zone: [", t1, "m, ", t2, "m]"),
       x = "Time (Months)", y = "Overall Survival S(t)") +
  theme(legend.position = "none")

# Panel C: Implied Hazards h(t)
p3 <- ggplot(plot_df, aes(x = time, y = hazard, color = model, linetype = model)) +
  geom_rect(aes(xmin = t1, xmax = t2, ymin = 0, ymax = max(h_ext) * 1.5),
            fill = "grey90", alpha = 0.015, color = NA, inherit.aes = FALSE) +
  geom_vline(xintercept = c(t1, t2), linetype = "dotted", color = "grey40") +
  geom_line(linewidth = 1) +
  coord_cartesian(ylim = c(0, 0.06)) +
  scale_color_manual(values = custom_colors) +
  scale_linetype_manual(values = custom_lines) +
  theme_minimal(base_size = 12) +
  labs(title = "C. Implied Hazard Rates h(t) (Check for Smoothness)",
       subtitle = "Continuous transition without kinks at boundaries",
       x = "Time (Months)", y = "Hazard Rate") +
  theme(legend.position = "none")

# Combine into a dashboard
dashboard <- p1 / (p2 | p3) + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom")

print(dashboard)