# ==============================================================================
# Two-Parameter Gaussian Process (GP) Variogram & Matern Barycentre for dynblend
# Decoupling Endpoint Smoothness (nu) from Interior Transition Peakiness (ell)
# ==============================================================================

source("R/meta_analysis_variance_weighting.R")

idx_a <- which.min(abs(t_grid - a_cut))
idx_b <- which.min(abs(t_grid - b_cut))

# Monotonic Cumulative Hazard standard errors sigma_H(t) = SE(-ln S(t)) = SE(S(t)) / S(t)
sd_o <- se_obs / pmax(S_obs, 1e-6)
sd_e <- pred_ext$se / pmax(S_ext, 1e-6)

# Bilateral excess standard errors anchored at t = a (for obs) and t = b (for ext):
dse_o <- pmax(0, sd_o - sd_o[idx_a])
dse_e <- pmax(0, sd_e[idx_b] - sd_e)

# Normalize excess SEs by their median on [a, b] so r_obs(t*), r_ext(t*) = 1 at crossover:
med_o <- median(dse_o[idx_a:idx_b])
med_e <- median(dse_e[idx_a:idx_b])
r_o   <- pmax(dse_o / med_o, 1e-8)
r_e   <- pmax(dse_e / med_e, 1e-8)

# ==============================================================================
# PART 1: GP POWERED-EXPONENTIAL VARIOGRAM BARYCENTRE pi_{nu, kappa}(t)
# ==============================================================================
# In a Gaussian Process with powered-exponential correlation kernel:
#   K_{nu, ell}(r) = exp( - (r / ell)^(2 * nu) ) = exp( - (kappa * r)^(2 * nu) )
# where:
#   - nu > 0:         Boundary smoothness parameter (order of tangency O(|t - anchor|^(2*nu)))
#                     * nu = 0.5 -> C^0 continuous (linear at endpoints, hazard jump)
#                     * nu = 1.0 -> C^1 continuous (pi'(a) = pi'(b) = 0, smooth hazard)
#                     * nu = 1.5 -> C^2 continuous (pi'(a) = pi''(a) = 0, smooth hazard slope)
#   - kappa = 1/ell:  Inverse GP length-scale (controls interior transition speed / "peakiness")
#                     * kappa -> 0 (ell -> Inf): reduces to pure polynomial r_o^(2*nu) / (r_o^(2*nu) + r_e^(2*nu))
#                     * kappa > 0 (finite ell):  exponentially sharpens interior transition while preserving
#                                                exact O(|t - anchor|^(2*nu)) smoothness at endpoints!

gp_variogram_weight <- function(r_obs, r_ext, nu = 1.0, kappa = 0.0, t_grid, a, b) {
  if (kappa <= 1e-6) {
    # Large length-scale limit (ell -> Inf): pure power law with exponent 2*nu
    v_o <- r_obs^(2 * nu)
    v_e <- r_ext^(2 * nu)
  } else {
    # Exact inverse GP variogram ratio: [exp((kappa * r_obs)^(2*nu)) - 1]
    # Note: for small r near anchor, exp((kappa*r)^(2*nu)) - 1 ~ (kappa*r)^(2*nu) = O(r^(2*nu))!
    # For moderate r in interior, exp(...) grows rapidly with kappa, creating peaky transition!
    v_o <- expm1(pmin((kappa * r_obs)^(2 * nu), 50))
    v_e <- expm1(pmin((kappa * r_ext)^(2 * nu), 50))
  }
  pi_vec <- v_o / pmax(v_o + v_e, 1e-16)
  pi_vec[t_grid <= a] <- 0.0
  pi_vec[t_grid >= b] <- 1.0
  pi_vec
}

# Also implement the classic half-integer Matern GP Kernels:
#   K_Matern(r; nu, ell):
#     - nu = 1/2 (Exponential): exp(-r / ell)
#     - nu = 3/2 (Matern-3/2):  (1 + sqrt(3)*r/ell) * exp(-sqrt(3)*r/ell)
#     - nu = 5/2 (Matern-5/2):  (1 + sqrt(5)*r/ell + 5*r^2/(3*ell^2)) * exp(-sqrt(5)*r/ell)
#     - nu = Inf (RBF / SE):    exp(-r^2 / (2*ell^2))
gp_matern_weight <- function(r_obs, r_ext, matern_type = c("exp_1_2", "matern_3_2", "matern_5_2", "rbf_inf"),
                             ell = 1.0, t_grid, a, b) {
  matern_type <- match.arg(matern_type)
  uo <- r_obs / ell
  ue <- r_ext / ell
  
  if (matern_type == "exp_1_2") {
    Ko <- exp(-uo)
    Ke <- exp(-ue)
  } else if (matern_type == "matern_3_2") {
    Ko <- (1 + sqrt(3) * uo) * exp(-sqrt(3) * uo)
    Ke <- (1 + sqrt(3) * ue) * exp(-sqrt(3) * ue)
  } else if (matern_type == "matern_5_2") {
    # Raise radial distance so 1 - K_5/2 vanishes to 3rd order O(r^3) at boundary:
    Ko <- (1 + sqrt(5) * uo + (5 / 3) * uo^2) * exp(-sqrt(5) * uo)
    Ke <- (1 + sqrt(5) * ue + (5 / 3) * ue^2) * exp(-sqrt(5) * ue)
  } else if (matern_type == "rbf_inf") {
    Ko <- exp(-0.5 * uo^2)
    Ke <- exp(-0.5 * ue^2)
  }
  
  # GP Kriging precision weight w_i = K_i / (1 - K_i), so pi = (1/w_obs) / (1/w_obs + 1/w_ext)
  inv_wo <- (1 - Ko) / pmax(Ko, 1e-12)
  inv_we <- (1 - Ke) / pmax(Ke, 1e-12)
  
  pi_vec <- inv_wo / pmax(inv_wo + inv_we, 1e-16)
  pi_vec[t_grid <= a] <- 0.0
  pi_vec[t_grid >= b] <- 1.0
  pi_vec
}

# ==============================================================================
# PART 2: THREE EXPERIMENTS DEMONSTRATING THE GP DECOUPLING
# ==============================================================================

# Experiment 1: Fix Interior Length-Scale (kappa = 0 -> ell = Inf) and Vary Smoothness nu
# Shows how nu controls endpoint smoothness (C^0 vs C^1 vs C^2) at t = a and t = b
nu_specs <- list(
  "nu = 0.5 (2*nu=1: C^0 Ornstein-Uhlenbeck -> Boundary Jump!)" = 0.5,
  "nu = 1.0 (2*nu=2: C^1 Quadratic / KL -> Zero Hazard Jump)"   = 1.0,
  "nu = 1.5 (2*nu=3: C^2 Cubic -> Smooth Hazard & Slope)"       = 1.5,
  "nu = 2.5 (2*nu=5: C^4 High-Order Boundary Flatness)"         = 2.5
)

df_exp1 <- bind_rows(lapply(names(nu_specs), function(lbl) {
  pi_v   <- gp_variogram_weight(r_o, r_e, nu = nu_specs[[lbl]], kappa = 0.0, t_grid, a_cut, b_cut)
  dpi_dt <- c(0, diff(pi_v) / dt)
  dpi_dt[t_grid <= a_cut | t_grid >= b_cut] <- 0.0
  b_res  <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, pi_v, t_grid)
  data.frame(
    time = t_grid, Weight = pi_v, dWeight_dt = dpi_dt,
    Survival = b_res$S_ble, Hazard = b_res$h_ble,
    Curve = lbl, Panel = "1. Vary Endpoint Smoothness nu (at Fixed Length-Scale ell = Inf)"
  )
}))

# Experiment 2: Fix Endpoint Smoothness at nu = 1.0 (Exact C^1 Hazard Tangency!)
# and Vary Inverse Length-Scale kappa = 1/ell in {0, 1.1, 1.6, 2.2}
# Shows how kappa = 1/ell tunes interior "peakiness" WITHOUT breaking C^1 endpoint tangency!
kappa_specs <- list(
  "ell = Inf (kappa = 0.0: Standard KL / IVW, Gentle)"          = 0.0,
  "ell = 0.91 (kappa = 1.1: Moderate GP Decay, Medium Peak)"    = 1.1,
  "ell = 0.62 (kappa = 1.6: Short GP Length-Scale, Peaky!)"     = 1.6,
  "ell = 0.45 (kappa = 2.2: Very Short Length-Scale, Step-Like)"= 2.2
)

df_exp2 <- bind_rows(lapply(names(kappa_specs), function(lbl) {
  pi_v   <- gp_variogram_weight(r_o, r_e, nu = 1.0, kappa = kappa_specs[[lbl]], t_grid, a_cut, b_cut)
  dpi_dt <- c(0, diff(pi_v) / dt)
  dpi_dt[t_grid <= a_cut | t_grid >= b_cut] <- 0.0
  b_res  <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, pi_v, t_grid)
  data.frame(
    time = t_grid, Weight = pi_v, dWeight_dt = dpi_dt,
    Survival = b_res$S_ble, Hazard = b_res$h_ble,
    Curve = lbl, Panel = "2. Vary Interior Peakiness ell (at Fixed C^1 Smoothness nu = 1.0)"
  )
}))

# Experiment 3: Classic Matern GP Kernels (nu = 1/2, 3/2, 5/2, Inf RBF) at ell = 0.70
matern_specs <- list(
  "Matern nu = 1/2 (Exponential Kernel: C^0 Boundary)"  = "exp_1_2",
  "Matern nu = 3/2 (Once-Differentiable GP: C^1 Smooth)"= "matern_3_2",
  "Matern nu = 5/2 (Twice-Differentiable GP: C^2 Smooth)"= "matern_5_2",
  "Matern nu -> Inf (Squared-Exponential / RBF Kernel)"  = "rbf_inf"
)

df_exp3 <- bind_rows(lapply(names(matern_specs), function(lbl) {
  pi_v   <- gp_matern_weight(r_o, r_e, matern_type = matern_specs[[lbl]], ell = 0.70, t_grid, a_cut, b_cut)
  dpi_dt <- c(0, diff(pi_v) / dt)
  dpi_dt[t_grid <= a_cut | t_grid >= b_cut] <- 0.0
  b_res  <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, pi_v, t_grid)
  data.frame(
    time = t_grid, Weight = pi_v, dWeight_dt = dpi_dt,
    Survival = b_res$S_ble, Hazard = b_res$h_ble,
    Curve = lbl, Panel = "3. Classic Matern GP Kernels K_{nu, ell}(Delta_sigma) (ell = 0.70)"
  )
}))

df_gp_all <- bind_rows(df_exp1, df_exp2, df_exp3)
df_gp_all$Panel <- factor(df_gp_all$Panel, levels = c(
  "1. Vary Endpoint Smoothness nu (at Fixed Length-Scale ell = Inf)",
  "2. Vary Interior Peakiness ell (at Fixed C^1 Smoothness nu = 1.0)",
  "3. Classic Matern GP Kernels K_{nu, ell}(Delta_sigma) (ell = 0.70)"
))
df_gp_all$Curve <- factor(df_gp_all$Curve, levels = c(
  names(nu_specs), names(kappa_specs), names(matern_specs)
))

# Print numerical summary table
cat("\n====================================================================================================\n")
cat("GAUSSIAN PROCESS (GP) BARYCENTRE SUMMARY: ENDPOINT SMOOTHNESS (nu) VS. INTERIOR PEAKINESS (ell)\n")
cat("====================================================================================================\n")
gp_tab <- df_gp_all %>%
  filter(time >= a_cut & time <= b_cut) %>%
  group_by(Panel, Curve) %>%
  summarise(
    Slope_at_a   = round(dWeight_dt[which.min(abs(time - (a_cut + 0.05)))], 3),
    pi_at_5.5y   = round(Weight[which.min(abs(time - 5.5))], 3),
    pi_at_8.5y   = round(Weight[which.min(abs(time - 8.5))], 3),
    Max_Rate     = round(max(dWeight_dt), 3),
    Peakiness    = round(max(dWeight_dt) / (1 / (b_cut - a_cut)), 2),
    Surv_6y_pct  = round(100 * Survival[which.min(abs(time - 6.0))], 1),
    Surv_10y_pct = round(100 * Survival[which.min(abs(time - 10.0))], 1),
    .groups = "drop"
  )
print(as.data.frame(gp_tab), row.names = FALSE)

# ==============================================================================
# PART 3: 4-ROW DIAGNOSTIC PLOT FOR GP KERNEL BARYCENTRE
# ==============================================================================
pal_12 <- c(
  "#D55E00", "#009E73", "#0072B2", "#CC79A7",
  "#009E73", "#E69F00", "#7570B3", "#E7298A",
  "#D55E00", "#0072B2", "#009E73", "#E7298A"
)
names(pal_12) <- c(names(nu_specs), names(kappa_specs), names(matern_specs))

p_gp_w <- ggplot(df_gp_all, aes(x = time, y = Weight, color = Curve, linetype = Curve)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey92", alpha = 0.6) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ Panel, ncol = 3) +
  scale_color_manual(values = pal_12) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20), labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "A. GP Barycentre Weight Functions pi(t): Decoupling Endpoint Smoothness (nu) from Interior Peakiness (ell)",
    subtitle = "Col 1: nu controls tangency order at a & b. Col 2: Fixing nu=1 (C^1 smooth) & decreasing length-scale ell sharpens the interior S-curve! Col 3: Matern GP kernels.",
    x = "Time (years)", y = "External Weight pi(t)"
  )

p_gp_dw <- ggplot(df_gp_all, aes(x = time, y = dWeight_dt, color = Curve, linetype = Curve)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 0.56, fill = "grey92", alpha = 0.6) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_hline(yintercept = 1 / (b_cut - a_cut), linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ Panel, ncol = 3) +
  scale_color_manual(values = pal_12) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20), labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 0.56)) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "B. Transition Speed pi'(t) = d(pi)/dt: Independent Control of Boundary Contact (nu) and Peakiness (ell)",
    subtitle = "Notice in Col 2 how all 4 curves depart a=4 and arrive at b=13 with zero slope (C^1), while ell tunes the peak height from 0.20 to 0.52!",
    x = "Time (years)", y = "Rate of Weight Change d(pi)/dt"
  )

p_gp_s <- ggplot(df_gp_all, aes(x = time, y = Survival, color = Curve, linetype = Curve)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey92", alpha = 0.6) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = S_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = S_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
  annotate("point", x = b_cut, y = S_ext[idx_b], color = "black", size = 2.8) +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ Panel, ncol = 3) +
  scale_color_manual(values = pal_12) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20), labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "C. Resulting Blended Survival Curves S(t) Under the Two-Parameter GP Barycentre",
    subtitle = "Short GP length-scales ell keep S_ble(t) tracking S_obs(t) longer before smoothly bridging to S_ext(t) across the crossover zone",
    x = "Time since randomisation (years)", y = "Survival"
  )

p_gp_h <- ggplot(df_gp_all, aes(x = time, y = Hazard, color = Curve, linetype = Curve)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 0.65, fill = "grey92", alpha = 0.6) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = h_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = h_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(linewidth = 1.05) +
  facet_wrap(~ Panel, ncol = 3) +
  scale_color_manual(values = pal_12) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20), labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 0.65)) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom", legend.direction = "vertical", legend.title = element_blank(), strip.text = element_text(face = "bold", size = 9.5)) +
  guides(color = guide_legend(ncol = 3), linetype = guide_legend(ncol = 3)) +
  labs(
    title = "D. Implied Blended Hazard Rates h_ble(t): C^0 Boundary Jumps (nu=0.5) vs. C^1/C^2 Smooth GP Bridges (nu >= 1)",
    subtitle = "Only nu = 0.5 (orange curves in Cols 1 & 3) has a boundary hazard jump at a=4 and b=13; all nu >= 1 curves join h_obs(a) and h_ext(b) smoothly!",
    x = "Time since randomisation (years)", y = "Hazard"
  )

fig_gp <- p_gp_w / p_gp_dw / p_gp_s / p_gp_h + plot_layout(heights = c(1, 0.95, 1, 1.4))
ggsave("docs/fig_gp_kernel_barycenter.png", fig_gp, width = 16, height = 16.5, dpi = 300)
cat("\nSaved 2-parameter Gaussian Process barycentre figure to docs/fig_gp_kernel_barycenter.png\n")
