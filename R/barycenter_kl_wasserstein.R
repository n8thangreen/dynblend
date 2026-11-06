# ==============================================================================
# Dynamic Survival Blending: Pointwise Barycenter Optimization (KL vs Wasserstein)
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
# PART 1: DISTRIBUTIONAL DISTANCE MEASURES (USING CURVE VARIANCES)
# ==============================================================================
# At each time t, each curve is represented by its point estimate (mu) and
# standard error (sd), i.e., epistemic distribution N(mu, sd^2) (or on H scale).

# 1. Reverse KL Divergence: D_KL(P_blend || P_anchor)
d_kl_reverse <- function(mu_blend, sd_blend, mu_anc, sd_anc) {
  var_b <- pmax(sd_blend^2, 1e-10)
  var_a <- pmax(sd_anc^2, 1e-10)
  0.5 * (log(var_a / var_b) + (var_b + (mu_blend - mu_anc)^2) / var_a - 1)
}

# 2. Forward KL Divergence: D_KL(P_anchor || P_blend)
d_kl_forward <- function(mu_blend, sd_blend, mu_anc, sd_anc) {
  var_b <- pmax(sd_blend^2, 1e-10)
  var_a <- pmax(sd_anc^2, 1e-10)
  0.5 * (log(var_b / var_a) + (var_a + (mu_anc - mu_blend)^2) / var_b - 1)
}

# 3. 2-Wasserstein Squared Distance (W_2^2)
d_wasserstein2 <- function(mu_blend, sd_blend, mu_anc, sd_anc) {
  (mu_blend - mu_anc)^2 + (sd_blend - sd_anc)^2
}

# 4. 1-Wasserstein Distance (W_1)
d_wasserstein1 <- function(mu_blend, sd_blend, mu_anc, sd_anc) {
  abs(mu_blend - mu_anc) + abs(sd_blend - sd_anc)
}

# ==============================================================================
# PART 2: POINTWISE BARYCENTER OPTIMIZER AT EACH TIME t
# ==============================================================================
eval_blend_state <- function(pi_ext, S_o, sd_o, S_e, sd_e,
                             scale = c("linear", "geometric"),
                             metric = c("kl_rev", "kl_fwd", "wass2", "wass1")) {
  scale  <- match.arg(scale)
  metric <- match.arg(metric)
  w_obs  <- 1 - pi_ext
  w_ext  <- pi_ext
  
  if (scale == "linear") {
    mu_o <- S_o; sd_o_use <- sd_o
    mu_e <- S_e; sd_e_use <- sd_e
  } else {
    S_o_safe <- pmax(S_o, 1e-6)
    S_e_safe <- pmax(S_e, 1e-6)
    mu_o     <- -log(S_o_safe)
    mu_e     <- -log(S_e_safe)
    sd_o_use <- sd_o / S_o_safe
    sd_e_use <- sd_e / S_e_safe
  }
  
  mu_b <- w_obs * mu_o + w_ext * mu_e
  
  # Monotonic barycentric uncertainty interpolation:
  # - KL barycenters interpolate precision: tau_b = (1-pi)*tau_o + pi*tau_e
  # - Wasserstein barycenters (McCann interpolant) interpolate SD: sd_b = (1-pi)*sd_o + pi*sd_e
  if (metric %in% c("kl_rev", "kl_fwd")) {
    tau_b <- w_obs / pmax(sd_o_use^2, 1e-10) + w_ext / pmax(sd_e_use^2, 1e-10)
    sd_b  <- 1 / sqrt(tau_b)
  } else {
    sd_b  <- w_obs * sd_o_use + w_ext * sd_e_use
  }
  
  list(mu = mu_b, sd = sd_b, mu_o = mu_o, sd_o = sd_o_use, mu_e = mu_e, sd_e = sd_e_use)
}

barycenter_objective <- function(pi_ext, S_o, sd_o, S_e, sd_e,
                                 lambda_t = 0.5,
                                 metric = c("kl_rev", "kl_fwd", "wass2", "wass1"),
                                 scale = c("linear", "geometric"),
                                 weight_mode = c("intrinsic_var", "prec_weighted", "sd_weighted")) {
  metric      <- match.arg(metric)
  scale       <- match.arg(scale)
  weight_mode <- match.arg(weight_mode)
  
  st <- eval_blend_state(pi_ext, S_o, sd_o, S_e, sd_e, scale = scale, metric = metric)
  
  if (metric == "kl_rev") {
    d_obs <- d_kl_reverse(st$mu, st$sd, st$mu_o, st$sd_o)
    d_ext <- d_kl_reverse(st$mu, st$sd, st$mu_e, st$sd_e)
  } else if (metric == "kl_fwd") {
    d_obs <- d_kl_forward(st$mu, st$sd, st$mu_o, st$sd_o)
    d_ext <- d_kl_forward(st$mu, st$sd, st$mu_e, st$sd_e)
  } else if (metric == "wass2") {
    d_obs <- d_wasserstein2(st$mu, st$sd, st$mu_o, st$sd_o)
    d_ext <- d_wasserstein2(st$mu, st$sd, st$mu_e, st$sd_e)
  } else if (metric == "wass1") {
    d_obs <- d_wasserstein1(st$mu, st$sd, st$mu_o, st$sd_o)
    d_ext <- d_wasserstein1(st$mu, st$sd, st$mu_e, st$sd_e)
  }
  
  if (weight_mode == "intrinsic_var") {
    w_o <- 1 - lambda_t
    w_e <- lambda_t
  } else if (weight_mode == "prec_weighted") {
    tau_o <- 1 / pmax(st$sd_o^2, 1e-10)
    tau_e <- 1 / pmax(st$sd_e^2, 1e-10)
    w_o   <- (1 - lambda_t) * tau_o
    w_e   <- lambda_t * tau_e
  } else if (weight_mode == "sd_weighted") {
    inv_sd_o <- 1 / pmax(st$sd_o, 1e-5)
    inv_sd_e <- 1 / pmax(st$sd_e, 1e-5)
    w_o      <- (1 - lambda_t) * inv_sd_o
    w_e      <- lambda_t * inv_sd_e
  }
  
  w_o * d_obs + w_e * d_ext
}

# Solve for optimal pi_ext(t) across the entire time grid via stats::optimize()
solve_barycenter_curve <- function(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext,
                                   t_grid, a, b,
                                   metric = c("kl_rev", "kl_fwd", "wass2", "wass1"),
                                   scale = c("geometric", "linear"),
                                   weight_mode = c("intrinsic_var", "prec_weighted", "sd_weighted"),
                                   use_smoothstep = TRUE) {
  metric      <- match.arg(metric)
  scale       <- match.arg(scale)
  weight_mode <- match.arg(weight_mode)
  
  n_t    <- length(t_grid)
  dt     <- diff(t_grid)[1]
  pi_opt <- numeric(n_t)
  
  for (i in seq_len(n_t)) {
    ti <- t_grid[i]
    if (ti <= a) {
      pi_opt[i] <- 0.0
    } else if (ti >= b) {
      pi_opt[i] <- 1.0
    } else {
      u <- (ti - a) / (b - a)
      lam_i <- if (use_smoothstep) (3 * u^2 - 2 * u^3) else 0.5
      
      opt_res <- optimize(
        f           = barycenter_objective,
        interval    = c(0, 1),
        S_o         = S_obs[i],
        sd_o        = se_obs[i],
        S_e         = S_ext[i],
        sd_e        = se_ext[i],
        lambda_t    = lam_i,
        metric      = metric,
        scale       = scale,
        weight_mode = weight_mode
      )
      pi_opt[i] <- opt_res$minimum
    }
  }
  
  # Derivative pi'(t)
  dpi <- c(diff(pi_opt) / dt, tail(diff(pi_opt) / dt, 1))
  
  if (scale == "geometric") {
    # Che et al. (2023) Eq (1): S_ble(t) = S_obs(t)^(1 - pi(t)) * S_ext(t)^pi(t)
    S_blend  <- (pmax(S_obs, 1e-8)^(1 - pi_opt)) * (pmax(S_ext, 1e-8)^pi_opt)
    H_obs    <- -log(pmax(S_obs, 1e-8))
    H_ext    <- -log(pmax(S_ext, 1e-8))
    # Analytical hazard from Che et al. (2023) p. 303:
    # h_ble(t) = (1 - pi(t))*h_obs(t) + pi(t)*h_ext(t) + pi'(t)*(H_ext(t) - H_obs(t))
    h_blend  <- (1 - pi_opt) * h_obs + pi_opt * h_ext + dpi * (H_ext - H_obs)
    h_blend  <- pmax(h_blend, 1e-8)
    
    # Standard error of S_blend
    se_H_obs <- se_obs / pmax(S_obs, 1e-6)
    se_H_ext <- se_ext / pmax(S_ext, 1e-6)
    se_H_ble <- sqrt((1 - pi_opt)^2 * se_H_obs^2 + pi_opt^2 * se_H_ext^2)
    se_blend <- S_blend * se_H_ble
  } else {
    # Linear mixture: S_ble(t) = (1 - pi(t))*S_obs(t) + pi(t)*S_ext(t)
    S_blend  <- (1 - pi_opt) * S_obs + pi_opt * S_ext
    se_blend <- sqrt((1 - pi_opt)^2 * se_obs^2 + pi_opt^2 * se_ext^2)
    dS       <- -c(diff(S_blend) / dt, tail(diff(S_blend) / dt, 1))
    h_blend  <- pmax(dS / pmax(S_blend, 1e-6), 0)
  }
  
  list(
    pi_ext   = pi_opt,
    S_blend  = S_blend,
    se_blend = se_blend,
    lcl      = pmax(S_blend - 1.96 * se_blend, 0),
    ucl      = pmin(S_blend + 1.96 * se_blend, 1),
    h_blend  = h_blend
  )
}

# ==============================================================================
# PART 3: REPRODUCE CHE ET AL. (2023) SETUP (FIGURES 2, 3, 4 & 6)
# ==============================================================================
# Time horizon: 0 to 20 years (matching Figure 2, 3, 4 in Che et al., 2023)
# - Follow-up period [0, a]: a = 4 years (48 months, CLL-8 trial with >70% censoring)
# - Blending interval [a, b]: [4, 13] years (matching Figure 2 & 3 where expert
#   elicitation anchors survival at ~10% at 13 years and ~1.3% at 15 years)
# - Long-term [b, T*]: [13, 20] years

t_grid <- seq(0.05, 20, by = 0.05)
dt     <- diff(t_grid)[1]
a_cut  <- 4.0   # End of trial follow-up (years)
b_cut  <- 13.0  # End of blending interval (years)

# 1. Simulate Short-Term Trial Data (matching Figure 2 & Figure 6 of Che et al., 2023:
#    N = 403, 4-year follow-up, ~80% survival at 4 years, >70% administrative censoring,
#    extrapolating to ~33% survival at 20 years)
set.seed(101)
n_trial  <- 403
t_events <- rweibull(n_trial, shape = 1.05, scale = 18.5)
c_admin  <- runif(n_trial, min = 3.2, max = 4.0)
df_obs   <- data.frame(
  time   = pmin(t_events, c_admin),
  status = as.numeric(t_events <= c_admin)
)

# Kaplan-Meier fit for plotting the observed trial curve on [0, a] (purple dashed in Fig 2)
km_fit <- survfit(Surv(time, status) ~ 1, data = df_obs)
df_km  <- data.frame(time = km_fit$time, surv = km_fit$surv)

# Fit Observed Data Model S_obs(t) (green dashed curve in Fig 2 & 6)
fit_obs  <- flexsurvreg(Surv(time, status) ~ 1, data = df_obs, dist = "weibull")
pred_obs <- summary(fit_obs, t = t_grid, type = "survival", se = TRUE)[[1]]
haz_obs  <- summary(fit_obs, t = t_grid, type = "hazard")[[1]]

S_obs    <- pred_obs$est
# Beyond follow-up (t > 4y), uncertainty in S_obs widens steadily (as in Fig 6 of Che et al.)
se_obs   <- pred_obs$se + ifelse(t_grid > a_cut, 0.007 * (t_grid - a_cut)^1.10, 0)
lcl_obs  <- pmax(S_obs - 1.96 * se_obs, 0)
ucl_obs  <- pmin(S_obs + 1.96 * se_obs, 1)
h_obs    <- haz_obs$est

# 2. Construct External Survival Curve S_ext(t) (blue dashed curve in Fig 2, 4, 5, 6):
#    Gompertz curve lying strictly below S_obs(t), with S_ext(4) ~ 66%, S_ext(10) ~ 24%,
#    and passing through the expert-elicited point S_ext(13) = 10% (black dot in Fig 2),
#    with h_ext(t) rising to ~0.60 at 20 years (matching Fig 4 of Che et al.)
set.seed(202)
n_ext_synth <- 300
t_ext_synth <- flexsurv::rgompertz(n_ext_synth, shape = 0.115, rate = 0.075)
df_ext      <- data.frame(time = t_ext_synth, status = rep(1, n_ext_synth))
fit_ext     <- flexsurvreg(Surv(time, status) ~ 1, data = df_ext, dist = "gompertz")
pred_ext    <- summary(fit_ext, t = t_grid, type = "survival", se = TRUE)[[1]]
haz_ext     <- summary(fit_ext, t = t_grid, type = "hazard")[[1]]

S_ext    <- pred_ext$est
se_ext   <- pmax(pred_ext$se, 0.012)
lcl_ext  <- pmax(S_ext - 1.96 * se_ext, 0)
ucl_ext  <- pmin(S_ext + 1.96 * se_ext, 1)
h_ext    <- haz_ext$est

cat(sprintf("Che et al. (2023) Calibration Check:\n"))
cat(sprintf("  * S_obs at 4y (end of follow-up): %.1f%% | at 13y: %.1f%% | at 15y: %.1f%%\n",
            100 * S_obs[which.min(abs(t_grid - 4))],
            100 * S_obs[which.min(abs(t_grid - 13))],
            100 * S_obs[which.min(abs(t_grid - 15))]))
cat(sprintf("  * S_ext at 4y:                    %.1f%% | at 13y: %.1f%% | at 15y: %.1f%%\n\n",
            100 * S_ext[which.min(abs(t_grid - 4))],
            100 * S_ext[which.min(abs(t_grid - 13))],
            100 * S_ext[which.min(abs(t_grid - 15))]))

# ==============================================================================
# PART 4: SOLVE BARYCENTERS (KL vs WASSERSTEIN) & COMPARE WITH CHE ET AL. BETA
# ==============================================================================

# 1. Che et al. Static Beta(3, 3) & Beta(2, 5) (from Figure 3 of Che et al., p. 303)
u_grid  <- pmin(pmax((t_grid - a_cut) / (b_cut - a_cut), 0), 1)
pi_b33  <- pbeta(u_grid, 3, 3)
pi_b25  <- pbeta(u_grid, 2, 5)
dpi_b33 <- dbeta(u_grid, 3, 3) / (b_cut - a_cut)
dpi_b25 <- dbeta(u_grid, 2, 5) / (b_cut - a_cut)

H_obs_vec <- -log(pmax(S_obs, 1e-8))
H_ext_vec <- -log(pmax(S_ext, 1e-8))

S_ble_b33  <- (S_obs^(1 - pi_b33)) * (S_ext^pi_b33)
h_ble_b33  <- (1 - pi_b33) * h_obs + pi_b33 * h_ext + dpi_b33 * (H_ext_vec - H_obs_vec)
se_ble_b33 <- S_ble_b33 * sqrt((1 - pi_b33)^2 * (se_obs / S_obs)^2 + pi_b33^2 * (se_ext / S_ext)^2)

S_ble_b25  <- (S_obs^(1 - pi_b25)) * (S_ext^pi_b25)
h_ble_b25  <- (1 - pi_b25) * h_obs + pi_b25 * h_ext + dpi_b25 * (H_ext_vec - H_obs_vec)
se_ble_b25 <- S_ble_b25 * sqrt((1 - pi_b25)^2 * (se_obs / S_obs)^2 + pi_b25^2 * (se_ext / S_ext)^2)

# 2. Pointwise Barycenter Optimizers (KL vs Wasserstein, Geometric & Linear)
res_kl_geo <- solve_barycenter_curve(
  S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, t_grid, a_cut, b_cut,
  metric = "kl_rev", scale = "geometric", weight_mode = "intrinsic_var"
)
res_kl_lin <- solve_barycenter_curve(
  S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, t_grid, a_cut, b_cut,
  metric = "kl_rev", scale = "linear", weight_mode = "intrinsic_var"
)
res_w2_tau_geo <- solve_barycenter_curve(
  S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, t_grid, a_cut, b_cut,
  metric = "wass2", scale = "geometric", weight_mode = "prec_weighted"
)
res_w2_sd_geo <- solve_barycenter_curve(
  S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, t_grid, a_cut, b_cut,
  metric = "wass2", scale = "geometric", weight_mode = "sd_weighted"
)
res_w2_sd_lin <- solve_barycenter_curve(
  S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, t_grid, a_cut, b_cut,
  metric = "wass2", scale = "linear", weight_mode = "sd_weighted"
)

# ==============================================================================
# PART 5: GENERATE FIGURES IN THE EXACT STYLE OF CHE ET AL. (2023)
# ==============================================================================

# --- FIGURE A: Direct Reproduction of Che et al. Figure 2 & Figure 6 Style ---
# Shows: Follow-up [0, a], Blending interval [a, b], Long-term [b, 20],
# Data fitting (green dashed + 95% ribbon), External info (blue dashed + 95% ribbon),
# Kaplan-Meier on [0, a] (purple dashed), and Blended curves (red/orange solid + 95% ribbon)

p_che_fig2 <- ggplot() +
  # Shaded 95% CI ribbons (matching Figure 6 of Che et al.)
  geom_ribbon(aes(x = t_grid, ymin = lcl_obs, ymax = ucl_obs), fill = "grey75", alpha = 0.35) +
  geom_ribbon(aes(x = t_grid, ymin = lcl_ext, ymax = ucl_ext), fill = "grey75", alpha = 0.35) +
  geom_ribbon(aes(x = t_grid, ymin = res_kl_geo$lcl, ymax = res_kl_geo$ucl), fill = "#F8766D", alpha = 0.22) +
  geom_ribbon(aes(x = t_grid, ymin = res_w2_sd_geo$lcl, ymax = res_w2_sd_geo$ucl), fill = "#E69F00", alpha = 0.22) +
  # Vertical boundary lines at a = 4 and b = 13 (matching Figure 2 of Che et al.)
  geom_vline(xintercept = a_cut, linetype = "dashed", color = "grey30", linewidth = 0.6) +
  geom_vline(xintercept = b_cut, linetype = "dotted", color = "grey30", linewidth = 0.6) +
  # Main survival curves
  geom_line(aes(x = t_grid, y = S_obs, color = "Data fitting (S_obs)", linetype = "Data fitting (S_obs)"), linewidth = 0.9) +
  geom_line(aes(x = t_grid, y = S_ext, color = "External info (S_ext)", linetype = "External info (S_ext)"), linewidth = 0.9) +
  geom_step(data = df_km, aes(x = time, y = surv, color = "Kaplan-Meier", linetype = "Kaplan-Meier"), linewidth = 0.9) +
  geom_line(aes(x = t_grid, y = S_ble_b33, color = "Blended: Che Beta(3,3)", linetype = "Blended: Che Beta(3,3)"), linewidth = 0.9) +
  geom_line(aes(x = t_grid, y = res_kl_geo$S_blend, color = "Blended: KL Barycenter", linetype = "Blended: KL Barycenter"), linewidth = 1.15) +
  geom_line(aes(x = t_grid, y = res_w2_sd_geo$S_blend, color = "Blended: Wasserstein Barycenter", linetype = "Blended: Wasserstein Barycenter"), linewidth = 1.15) +
  # Elicited expert point at 13y (~10% survival, black dot in Figure 2)
  geom_point(aes(x = b_cut, y = S_ext[which.min(abs(t_grid - b_cut))]), color = "black", size = 3) +
  # Region annotations matching Figure 2 of Che et al.
  annotate("text", x = 2.0, y = 0.05, label = "<- Follow-up ->", size = 3.5, color = "grey20") +
  annotate("text", x = 8.5, y = 0.96, label = "<- Blending interval ->", size = 3.5, color = "grey20") +
  annotate("text", x = 16.5, y = 0.96, label = "<- Long-term ->", size = 3.5, color = "grey20") +
  scale_color_manual(
    name = "model",
    values = c(
      "Data fitting (S_obs)"            = "#7CAE00", # Green as in Che Fig 2 & 6
      "External info (S_ext)"           = "#00BFC4", # Cyan/Blue as in Che Fig 2 & 6
      "Kaplan-Meier"                    = "#C77CFF", # Purple as in Che Fig 2
      "Blended: Che Beta(3,3)"          = "grey35",
      "Blended: KL Barycenter"          = "#F8766D", # Red/Coral as in Che Fig 2 & 6
      "Blended: Wasserstein Barycenter" = "#D95F02"  # Deep Orange
    )
  ) +
  scale_linetype_manual(
    name = "model",
    values = c(
      "Data fitting (S_obs)"            = "twodash",
      "External info (S_ext)"           = "dashed",
      "Kaplan-Meier"                    = "longdash",
      "Blended: Che Beta(3,3)"          = "dotted",
      "Blended: KL Barycenter"          = "solid",
      "Blended: Wasserstein Barycenter" = "solid"
    )
  ) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(), legend.position = "right") +
  labs(
    title = "Figure 2 / Figure 6 Analog: Blended Survival Curves (KL vs. Wasserstein Barycenters)",
    subtitle = "Follow-up [0, a=4y], Blending interval [a=4y, b=13y], Long-term [13y, 20y] with 95% credible/confidence ribbons",
    x = "Time since randomisation (years)",
    y = "Survival"
  )

# --- FIGURE B: Weight Functions pi(t) in the Style of Che et al. Figure 3 ---
df_weights_che <- data.frame(
  time = t_grid,
  `Che Static Beta(3,3)`             = pi_b33,
  `Che Static Beta(2,5)`             = pi_b25,
  `KL-Rev Barycenter (Geometric)`    = res_kl_geo$pi_ext,
  `KL-Rev Barycenter (Linear)`       = res_kl_lin$pi_ext,
  `Wasserstein (tau-Weighted, Geom)` = res_w2_tau_geo$pi_ext,
  `Wasserstein (1/sd-Weighted, Geom)`= res_w2_sd_geo$pi_ext,
  check.names = FALSE
) %>%
  pivot_longer(-time, names_to = "Weight_Function", values_to = "Weight")

p_che_fig3 <- ggplot(df_weights_che, aes(x = time, y = Weight, color = Weight_Function, linetype = Weight_Function)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey85", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(linewidth = 1) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank()) +
  labs(
    title = "Figure 3 Analog: Weight Functions pi(t) in Blending Interval [a=4y, b=13y]",
    subtitle = "KL & tau-Wasserstein shift weight earlier like Beta(2,5) as trial variance expands",
    x = "Time (years)",
    y = "Weight pi(t)"
  )

# --- FIGURE C: Blended Hazard Functions h_ble(t) in the Style of Che et al. Figure 4 ---
df_haz_che <- data.frame(
  time = rep(t_grid, 6),
  Hazard = c(
    h_obs, h_ext, h_ble_b33,
    res_kl_geo$h_blend, res_w2_sd_geo$h_blend, res_w2_sd_lin$h_blend
  ),
  Model = factor(rep(c(
    "Data fitting (h_obs)",
    "External info (h_ext)",
    "Blended: Che Beta(3,3)",
    "Blended: KL Barycenter (Geom)",
    "Blended: Wasserstein Barycenter (Geom)",
    "Blended: Wasserstein Barycenter (Linear)"
  ), each = length(t_grid)),
  levels = c(
    "Data fitting (h_obs)",
    "External info (h_ext)",
    "Blended: Che Beta(3,3)",
    "Blended: KL Barycenter (Geom)",
    "Blended: Wasserstein Barycenter (Geom)",
    "Blended: Wasserstein Barycenter (Linear)"
  ))
)

p_che_fig4 <- ggplot(df_haz_che, aes(x = time, y = Hazard, color = Model, linetype = Model)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 0.65, fill = "grey92", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(linewidth = 0.95) +
  scale_color_manual(
    name = "model",
    values = c(
      "Data fitting (h_obs)"                     = "#7CAE00",
      "External info (h_ext)"                    = "#00BFC4",
      "Blended: Che Beta(3,3)"                   = "grey35",
      "Blended: KL Barycenter (Geom)"            = "#F8766D",
      "Blended: Wasserstein Barycenter (Geom)"   = "#D95F02",
      "Blended: Wasserstein Barycenter (Linear)" = "#7570B3"
    )
  ) +
  scale_linetype_manual(
    name = "model",
    values = c(
      "Data fitting (h_obs)"                     = "twodash",
      "External info (h_ext)"                    = "dashed",
      "Blended: Che Beta(3,3)"                   = "dotted",
      "Blended: KL Barycenter (Geom)"            = "solid",
      "Blended: Wasserstein Barycenter (Geom)"   = "solid",
      "Blended: Wasserstein Barycenter (Linear)" = "longdash"
    )
  ) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 0.62)) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank()) +
  labs(
    title = "Figure 4 Analog: Blended Hazard Functions h_ble(t)",
    subtitle = "Turning-point hump in [a, b] from pi'(t)*(H_ext - H_obs) before merging into external hazard at b=13y",
    x = "Time since randomisation (years)",
    y = "Hazard"
  )

# --- FIGURE D: Linear vs. Geometric Barycenter Survival Comparison ---
df_lin_vs_geo <- data.frame(
  time = rep(t_grid, 6),
  Survival = c(
    S_obs, S_ext,
    res_kl_geo$S_blend, res_kl_lin$S_blend,
    res_w2_sd_geo$S_blend, res_w2_sd_lin$S_blend
  ),
  Curve = factor(rep(c(
    "Data fitting (S_obs)", "External info (S_ext)",
    "KL Barycenter (Geometric / Hazard Scale)", "KL Barycenter (Linear / Survival Scale)",
    "Wasserstein Barycenter (Geometric)", "Wasserstein Barycenter (Linear)"
  ), each = length(t_grid)),
  levels = c(
    "Data fitting (S_obs)", "External info (S_ext)",
    "KL Barycenter (Geometric / Hazard Scale)", "KL Barycenter (Linear / Survival Scale)",
    "Wasserstein Barycenter (Geometric)", "Wasserstein Barycenter (Linear)"
  ))
)

p_lin_vs_geo <- ggplot(df_lin_vs_geo, aes(x = time, y = Survival, color = Curve, linetype = Curve)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey92", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(linewidth = 0.95) +
  scale_color_manual(values = c(
    "Data fitting (S_obs)"                      = "#7CAE00",
    "External info (S_ext)"                     = "#00BFC4",
    "KL Barycenter (Geometric / Hazard Scale)"  = "#F8766D",
    "KL Barycenter (Linear / Survival Scale)"   = "#F8766D",
    "Wasserstein Barycenter (Geometric)"        = "#D95F02",
    "Wasserstein Barycenter (Linear)"           = "#D95F02"
  )) +
  scale_linetype_manual(values = c(
    "Data fitting (S_obs)"                      = "twodash",
    "External info (S_ext)"                     = "dashed",
    "KL Barycenter (Geometric / Hazard Scale)"  = "solid",
    "KL Barycenter (Linear / Survival Scale)"   = "dotted",
    "Wasserstein Barycenter (Geometric)"        = "solid",
    "Wasserstein Barycenter (Linear)"           = "dotted"
  )) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank()) +
  labs(
    title = "Linear (Arithmetic) vs. Geometric (Hazard) Barycenter Blending",
    subtitle = "Solid = Geometric (Che et al. Eq 1); Dotted = Linear mixture (1-pi)*S_obs + pi*S_ext",
    x = "Time since randomisation (years)",
    y = "Survival"
  )

# Save full Che et al. style dashboard
fig_che_dashboard <- p_che_fig2 / (p_che_fig3 | p_che_fig4) / p_lin_vs_geo +
  plot_layout(heights = c(1.35, 1, 1))

ggsave("docs/fig_barycenter_kl_vs_wasserstein.png", fig_che_dashboard, width = 13, height = 14, dpi = 300)
cat("Saved Che et al. (2023) style dashboard to docs/fig_barycenter_kl_vs_wasserstein.png\n\n")

# Print 20-year RMST summary table
calc_rmst20 <- function(S_vec) sum(S_vec) * dt

rmst_table <- data.frame(
  Method = c(
    "Data fitting only (S_obs, unadjusted)",
    "External info only (S_ext)",
    "Che et al. Static Beta(3,3) [Geometric]",
    "Che et al. Static Beta(2,5) [Geometric]",
    "KL-Reverse Barycenter [Geometric]",
    "KL-Reverse Barycenter [Linear]",
    "Wasserstein (tau-Weighted) Barycenter [Geometric]",
    "Wasserstein (1/sd-Weighted) Barycenter [Geometric]",
    "Wasserstein (1/sd-Weighted) Barycenter [Linear]"
  ),
  RMST_20yr = c(
    calc_rmst20(S_obs),
    calc_rmst20(S_ext),
    calc_rmst20(S_ble_b33),
    calc_rmst20(S_ble_b25),
    calc_rmst20(res_kl_geo$S_blend),
    calc_rmst20(res_kl_lin$S_blend),
    calc_rmst20(res_w2_tau_geo$S_blend),
    calc_rmst20(res_w2_sd_geo$S_blend),
    calc_rmst20(res_w2_sd_lin$S_blend)
  ),
  Surv_8yr_Pct = 100 * c(
    S_obs[which.min(abs(t_grid - 8))],
    S_ext[which.min(abs(t_grid - 8))],
    S_ble_b33[which.min(abs(t_grid - 8))],
    S_ble_b25[which.min(abs(t_grid - 8))],
    res_kl_geo$S_blend[which.min(abs(t_grid - 8))],
    res_kl_lin$S_blend[which.min(abs(t_grid - 8))],
    res_w2_tau_geo$S_blend[which.min(abs(t_grid - 8))],
    res_w2_sd_geo$S_blend[which.min(abs(t_grid - 8))],
    res_w2_sd_lin$S_blend[which.min(abs(t_grid - 8))]
  ),
  Surv_13yr_Pct = 100 * c(
    S_obs[which.min(abs(t_grid - 13))],
    S_ext[which.min(abs(t_grid - 13))],
    S_ble_b33[which.min(abs(t_grid - 13))],
    S_ble_b25[which.min(abs(t_grid - 13))],
    res_kl_geo$S_blend[which.min(abs(t_grid - 13))],
    res_kl_lin$S_blend[which.min(abs(t_grid - 13))],
    res_w2_tau_geo$S_blend[which.min(abs(t_grid - 13))],
    res_w2_sd_geo$S_blend[which.min(abs(t_grid - 13))],
    res_w2_sd_lin$S_blend[which.min(abs(t_grid - 13))]
  )
)

cat("====================================================================\n")
cat("20-YEAR SURVIVAL & RMST SUMMARY (CHE ET AL. 2023 CLL-8 CASE STUDY)\n")
cat("====================================================================\n")
print(rmst_table, digits = 3, row.names = FALSE)
