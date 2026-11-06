# ==============================================================================
# Unified Comparison: Original Entropy/Barycenter Methods vs. Meta-Analytic &
# "Peaky" (Slow -> Fast -> Slow) Exponential-Precision / Softmax S-Curve Methods
# Under Bilateral Excess Uncertainty (Delta_sigma_obs at a, Delta_sigma_ext at b)
# ==============================================================================

source("R/meta_analysis_variance_weighting.R")

idx_a <- which.min(abs(t_grid - a_cut))
idx_b <- which.min(abs(t_grid - b_cut))

# Monotonic Cumulative Hazard standard errors sigma_H(t) = SE(-ln S(t)) = SE(S(t)) / S(t)
sd_o <- se_obs / pmax(S_obs, 1e-6)
sd_e <- pred_ext$se / pmax(S_ext, 1e-6)

# Bilateral excess standard errors anchored at t = a (for obs) and t = b (for ext):
#   Delta_sigma_obs(t) = sigma_obs(t) - sigma_obs(a)   [vanishes at t = a]
#   Delta_sigma_ext(t) = sigma_ext(b) - sigma_ext(t)   [vanishes at t = b]
dse_o <- pmax(0, sd_o - sd_o[idx_a])
dse_e <- pmax(0, sd_e[idx_b] - sd_e)

# Bilateral Linear Excess Variance (O(|t - anchor|^1)):
dv_o_lin <- pmax(0, sd_o^2 - sd_o[idx_a]^2)
dv_e_lin <- pmax(0, sd_e[idx_b]^2 - sd_e^2)

# ==============================================================================
# PART 1: BILATERAL EXCESS VARIANCE COMPARISON (ONE-SIDED VS LINEAR VS QUADRATIC)
# ==============================================================================
eval_bilateral_delta <- function(dse_o_in, dse_e_in, is_squared_se = TRUE, unclamped_b = FALSE, regime_label) {
  if (is_squared_se) {
    v_o <- dse_o_in^2
    v_e <- dse_e_in^2
  } else {
    v_o <- dse_o_in
    v_e <- dse_e_in
  }
  tau_het_sq_loc <- pmax(0, (log(S_obs) - log(S_ext))^2 - (sd_o^2 + sd_e^2))
  
  pi_ivw   <- v_o / pmax(v_o + v_e, 1e-16)
  pi_p15   <- (v_o^0.75) / pmax((v_o^0.75) + (v_e^0.75), 1e-16)
  pi_p30   <- (v_o^1.5) / pmax((v_o^1.5) + (v_e^1.5), 1e-16)
  pi_re    <- v_o / pmax(v_o + v_e * (1 + 1.5 * tau_het_sq_loc), 1e-16)
  k_boltz  <- 1.5 / median(v_o[idx_a:idx_b] + v_e[idx_a:idx_b], na.rm = TRUE)
  bo       <- 1 - exp(-k_boltz * v_o)
  be       <- 1 - exp(-k_boltz * v_e)
  pi_boltz <- bo / pmax(bo + be, 1e-16)
  v0       <- median(v_o[idx_a:idx_b] + v_e[idx_a:idx_b], na.rm = TRUE) * 0.08
  lo       <- log(1 + v_o / v0)
  le       <- log(1 + v_e / v0)
  pi_log   <- lo / pmax(lo + le, 1e-16)
  
  w_list <- list(
    "1. Fixed-Effect IVW: 1/(Delta sigma)^2 [p=2]"               = pi_ivw,
    "2. Tempered Fractional Power: 1/(Delta sigma)^1.5 [p=1.5]"  = pi_p15,
    "3. Steep Power Barycenter: 1/(Delta sigma)^3 [p=3]"         = pi_p30,
    "4. Random-Effects (DerSimonian-Laird Heterogeneity)"        = pi_re,
    "5. Exponential / Boltzmann (1 - exp(-k*(Delta sigma)^2))"   = pi_boltz,
    "6. Inverse Log-Variance (1 / ln(1 + (Delta sigma)^2/s0^2))" = pi_log
  )
  
  bind_rows(lapply(names(w_list), function(m) {
    pi_vec <- w_list[[m]]
    pi_vec[t_grid <= a_cut] <- 0.0
    if (!unclamped_b) pi_vec[t_grid >= b_cut] <- 1.0
    b_res <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, pi_vec, t_grid)
    data.frame(
      time = t_grid, Weight = pi_vec, Survival = b_res$S_ble,
      Hazard = b_res$h_ble, Method = m, Regime = regime_label
    )
  }))
}

dv_e_const   <- rep(sd_e[idx_b]^2 - sd_e[idx_a]^2, length(t_grid))
df_onesided  <- eval_bilateral_delta(dv_o_lin, dv_e_const, FALSE, TRUE, "1. One-Sided Delta(sigma_obs^2) at a Only")
df_bilat_lin <- eval_bilateral_delta(dv_o_lin, dv_e_lin, FALSE, FALSE, "2. Bilateral Linear: Delta(sigma_obs^2) at a & Delta(sigma_ext^2) at b")
df_bilat_quad<- eval_bilateral_delta(dse_o, dse_e, TRUE, FALSE, "3. Bilateral Quadratic: (Delta sigma_obs)^2 at a & (Delta sigma_ext)^2 at b")

df_bilat_all <- bind_rows(df_onesided, df_bilat_lin, df_bilat_quad)
df_bilat_all$Regime <- factor(df_bilat_all$Regime, levels = c(
  "1. One-Sided Delta(sigma_obs^2) at a Only",
  "2. Bilateral Linear: Delta(sigma_obs^2) at a & Delta(sigma_ext^2) at b",
  "3. Bilateral Quadratic: (Delta sigma_obs)^2 at a & (Delta sigma_ext)^2 at b"
))

p_bilat_w <- ggplot(df_bilat_all, aes(x = time, y = Weight, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey90", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(linewidth = 1.05) +
  facet_wrap(~ Regime, ncol = 3) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20), labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "A. Weight Schedules pi(t): One-Sided vs Bilateral Excess Variance (Zero Splines Used!)",
    subtitle = "Panel 1 (one-sided at a) is concave and < 1 at b; Panel 2 (bilateral linear) reaches 1 at b; Panel 3 (bilateral squared Delta SE) forms a natural C^1 S-curve!",
    x = "Time (years)", y = "External Weight pi(t)"
  )

p_bilat_s <- ggplot(df_bilat_all, aes(x = time, y = Survival, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey90", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = S_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = S_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
  annotate("point", x = b_cut, y = S_ext[idx_b], color = "black", size = 2.8) +
  geom_line(linewidth = 1.05) +
  facet_wrap(~ Regime, ncol = 3) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20), labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "B. Blended Survival Curves S(t) Across One-Sided and Bilateral Excess-Variance Regimes",
    subtitle = "Bilateral excess variance (Panels 2 & 3) converges exactly to S_ext(b) at b=13y while keeping wide method separation",
    x = "Time since randomisation (years)", y = "Survival"
  )

p_bilat_h <- ggplot(df_bilat_all, aes(x = time, y = Hazard, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 0.62, fill = "grey90", alpha = 0.5) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = h_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = h_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(linewidth = 1.0) +
  facet_wrap(~ Regime, ncol = 3) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20), labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 0.58)) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom", legend.direction = "vertical", legend.title = element_blank(), strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "C. Implied Blended Hazard Rates h_ble(t): Why Squared Excess SE (Panel 3) Eliminates Boundary Jumps",
    subtitle = "Panel 2 has non-zero slope pi'(a) > 0 and pi'(b) > 0 (causing boundary hazard kinks); Panel 3 has pi'(a) = pi'(b) = 0 (C^1 smooth hazards!)",
    x = "Time since randomisation (years)", y = "Hazard"
  )

fig_bilateral <- p_bilat_w / p_bilat_s / p_bilat_h + plot_layout(heights = c(1, 1, 1.25))
ggsave("docs/fig_bilateral_excess_variance.png", fig_bilateral, width = 16, height = 13, dpi = 300)
cat("\nSaved bilateral excess-variance comparison to docs/fig_bilateral_excess_variance.png\n")

# Normalize each excess SE by its median over [a, b] so the crossover point t* is clean
# and dimensionless excess uncertainty ratios r(t) = dse_o(t) / dse_e(t) are well-scaled:
med_o <- median(dse_o[idx_a:idx_b])
med_e <- median(dse_e[idx_a:idx_b])
x_o   <- pmax(dse_o / med_o, 1e-8)
x_e   <- pmax(dse_e / med_e, 1e-8)

# ==============================================================================
# PART 1: DEFINE ALL 8 METHODS (ORIGINAL ENTROPY + META-ANALYTIC + PEAKY S-CURVE)
# ==============================================================================

# 1. Original BMA Differential Entropy (gamma = 1: exp(-h_diff) = 1 / Delta_sigma)
#    [Equivalent to 1/sigma Wasserstein Barycenter: flattest, non-zero slope at a and b]
pi_ent_bma <- x_o / (x_o + x_e)

# 2. Original Bernoulli Survival Shannon Entropy x Excess CI Width
#    H_Bern(S) = -S ln(S) - (1-S) ln(1-S), combined with excess SE:
H_bern_o <- -(S_obs * log(pmax(S_obs, 1e-8)) + (1 - S_obs) * log(pmax(1 - S_obs, 1e-8)))
H_bern_e <- -(S_ext * log(pmax(S_ext, 1e-8)) + (1 - S_ext) * log(pmax(1 - S_ext, 1e-8)))
eb_o     <- (H_bern_o * x_o)^2
eb_e     <- (H_bern_e * x_e)^2
pi_ent_bern <- eb_o / pmax(eb_o + eb_e, 1e-16)

# 3. Reverse-KL Barycenter / Fixed-Effect IVW (gamma = 2: 1 / (Delta_sigma)^2)
#    [Quadratic O((t-a)^2) at boundaries -> mild S-curve]
pi_kl_rev <- (x_o^2) / pmax(x_o^2 + x_e^2, 1e-16)

# 4. Forward-KL Barycenter (Moment-Matching with Mean-Separation Penalty)
#    Pointwise numerical optimization of Forward-KL divergence on [a, b]
H_obs_vec <- -log(pmax(S_obs, 1e-8))
H_ext_vec <- -log(pmax(S_ext, 1e-8))
pi_kl_fwd <- numeric(length(t_grid))
for (k in seq_along(t_grid)) {
  if (t_grid[k] <= a_cut) {
    pi_kl_fwd[k] <- 0.0
  } else if (t_grid[k] >= b_cut) {
    pi_kl_fwd[k] <- 1.0
  } else {
    vo <- pmax((dse_o[k])^2, 1e-10)
    ve <- pmax((dse_e[k])^2, 1e-10)
    # Outer precision weights + Forward-KL variance inflation from curve separation
    gap_sq <- (H_obs_vec[k] - H_ext_vec[k])^2
    # Forward KL damps the transition when gap_sq is large relative to variance
    pi_kl_fwd[k] <- vo / (vo + ve * (1 + 0.35 * gap_sq / (vo + ve + 0.05)))
  }
}

# 5. DerSimonian-Laird Random-Effects Meta-Analysis (Heterogeneity-Tempered)
tau_het_sq <- pmax(0, (H_obs_vec - H_ext_vec)^2 - (sd_o^2 + sd_e^2))
pi_re <- (x_o^2) / pmax(x_o^2 + (x_e^2) * (1 + 1.8 * tau_het_sq), 1e-16)

# 6.Inverse Log-Variance (1 / ln(1 + (Delta_sigma)^2 / s0^2))
pi_logvar <- log(1 + (x_o^2) / 0.15) / pmax(log(1 + (x_o^2) / 0.15) + log(1 + (x_e^2) / 0.15), 1e-16)

# 7. Tempered Entropy Softmax (beta = 5: w_i = exp(-5 * h_diff(Delta_sigma_i)) = 1 / (Delta_sigma_i)^5)
#    Logit(pi(t)) = 5 * [h_diff(Delta_sigma_obs(t)) - h_diff(Delta_sigma_ext(t))]
#    -> High-order O((t-a)^5) flatness near a and b: SLOW -> FAST -> SLOW!
pi_ent_soft6 <- (x_o^5) / pmax(x_o^5 + x_e^5, 1e-16)

# 8. Gibbs C^infty Essential-Flatness Precision Switch: w_i = exp(-kappa / Delta_sigma_i)
#    All derivatives vanish to infinite order at a and b (exp(-kappa / x) -> 0)!
#    With kappa = 2.2 on 1/Delta_sigma_i, it is strictly unimodal in dpi/dt:
#    stays flat at 0, switches rapidly across the crossover point, then locks flat at 1!
kappa_gibbs <- 2.2
go <- exp(-kappa_gibbs / pmax(x_e, 1e-6)) # weight on obs goes to 0 as x_e -> 0 at b
ge <- exp(-kappa_gibbs / pmax(x_o, 1e-6)) # weight on ext goes to 0 as x_o -> 0 at a
pi_gibbs <- ge / pmax(go + ge, 1e-16)

methods_all <- list(
  "1. BMA Diff. Entropy / W2 (1/Delta_sigma, beta=1) [Fastest Start, Linear]"        = pi_ent_bma,
  "2. Inverse Log-Variance (1/ln(1 + Delta_sigma^2)) [Gradual]"                      = pi_logvar,
  "3. Reverse-KL / Fixed-Effect IVW (1/Delta_sigma^2, beta=2) [Mild S-Curve]"        = pi_kl_rev,
  "4. Forward-KL Barycenter (Moment-Matching) [Conservative S-Curve]"                = pi_kl_fwd,
  "5. Bernoulli Survival Entropy x Delta_sigma (beta=2) [Mid-Survival Shift]"        = pi_ent_bern,
  "6. Random-Effects (DerSimonian-Laird Heterogeneity) [Delayed Shift]"              = pi_re,
  "7. Softmax Entropy Barycenter (exp(-5*h_diff), beta=5) [Peaky: Slow->Fast->Slow]" = pi_ent_soft6,
  "8. Gibbs C^inf Precision Switch (exp(-k/Delta_sigma)) [Peaky: Slow->Fast->Slow]"  = pi_gibbs
)

df_all_methods <- bind_rows(lapply(names(methods_all), function(m) {
  pi_vec <- methods_all[[m]]
  pi_vec[t_grid <= a_cut] <- 0.0
  pi_vec[t_grid >= b_cut] <- 1.0
  
  # Compute transition rate dpi/dt (peaky bell shape = slow -> fast -> slow)
  dpi_dt <- c(0, diff(pi_vec) / dt)
  dpi_dt[t_grid <= a_cut | t_grid >= b_cut] <- 0.0
  
  b_res <- blend_che_curve(S_obs, se_obs, h_obs, S_ext, se_ext, h_ext, pi_vec, t_grid)
  
  is_peaky <- grepl("beta=5|Gibbs", m)
  family_grp <- ifelse(
    grepl("beta=1|Log-Variance", m), "A. Low-Order / Gradual (beta <= 1: Fast Initial Change, Flat Middle)",
    ifelse(
      grepl("beta=5|Gibbs", m), "C. High-Order Entropy Softmax & Gibbs (Peaky: Slow -> Fast -> Slow!)",
      "B. Quadratic KL / Bernoulli / Random-Effects (beta = 2: Moderate S-Curve)"
    )
  )
  
  data.frame(
    time       = t_grid,
    Weight     = pi_vec,
    dWeight_dt = dpi_dt,
    Survival   = b_res$S_ble,
    Hazard     = b_res$h_ble,
    Method     = m,
    Family     = family_grp,
    IsPeaky    = is_peaky
  )
}))

df_all_methods$Method <- factor(df_all_methods$Method, levels = names(methods_all))
df_all_methods$Family <- factor(df_all_methods$Family, levels = c(
  "A. Low-Order / Gradual (beta <= 1: Fast Initial Change, Flat Middle)",
  "B. Quadratic KL / Bernoulli / Random-Effects (beta = 2: Moderate S-Curve)",
  "C. High-Order Entropy Softmax & Gibbs (Peaky: Slow -> Fast -> Slow!)"
))

# Print numerical diagnostic summary showing the Peakiness Ratio (Max dpi/dt vs Mean dpi/dt)
cat("\n====================================================================================================\n")
cat("COMPARISON OF TRANSITION DYNAMICS: GRADUAL VS. PEAKY (SLOW -> FAST -> SLOW) ENTROPY & META METHODS\n")
cat("====================================================================================================\n")
summary_dyn <- df_all_methods %>%
  filter(time >= a_cut & time <= b_cut) %>%
  group_by(Method) %>%
  summarise(
    pi_at_5.5y   = round(Weight[which.min(abs(time - 5.5))], 3),
    pi_at_8.5y   = round(Weight[which.min(abs(time - 8.5))], 3),
    pi_at_11.5y  = round(Weight[which.min(abs(time - 11.5))], 3),
    Max_Slope    = round(max(dWeight_dt), 3),
    Peakiness    = round(max(dWeight_dt) / (1 / (b_cut - a_cut)), 2),
    Surv_6y_pct  = round(100 * Survival[which.min(abs(time - 6.0))], 1),
    Surv_10y_pct = round(100 * Survival[which.min(abs(time - 10.0))], 1),
    .groups = "drop"
  )
print(as.data.frame(summary_dyn), row.names = FALSE)

# ==============================================================================
# PART 2: 4-ROW DIAGNOSTIC FIGURE (WEIGHT pi(t), SPEED pi'(t), SURVIVAL, HAZARD)
# ==============================================================================
cols_8 <- c(
  "#E69F00", "#56B4E9", "#009E73", "#0072B2",
  "#CC79A7", "#D55E00", "#7570B3", "#E7298A"
)
names(cols_8) <- names(methods_all)

p_row1_w <- ggplot(df_all_methods, aes(x = time, y = Weight, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey92", alpha = 0.6) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ Family, ncol = 3) +
  scale_color_manual(values = cols_8) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "A. Weight Schedules pi(t): From Gradual (Left) to Moderate S-Curve (Center) to Peaky 'Slow -> Fast -> Slow' (Right)",
    subtitle = "Why methods looked similar before: p=1..2 powers only change slope mildly. High-order Entropy Softmax (p=6) and Gibbs exp(-k/Delta_sigma^2) stay flat, switch fast, then flatten!",
    x = "Time (years)",
    y = "External Weight pi(t)"
  )

p_row2_dw <- ggplot(df_all_methods, aes(x = time, y = dWeight_dt, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 0.55, fill = "grey92", alpha = 0.6) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_hline(yintercept = 1 / (b_cut - a_cut), linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ Family, ncol = 3) +
  scale_color_manual(values = cols_8) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 0.55)) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "B. Transition Speed pi'(t) = d(pi)/dt ('Peakiness' of the Blending Rate Across [a=4y, b=13y])",
    subtitle = "Left: U-shaped/flat rate (changes immediately at a). Center: gentle bell curve. Right: strongly PEAKY bell curve (0 near a -> sharp peak at 7.5y -> 0 near b)!",
    x = "Time (years)",
    y = "Rate of Weight Change d(pi)/dt"
  )

p_row3_s <- ggplot(df_all_methods, aes(x = time, y = Survival, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 1, fill = "grey92", alpha = 0.6) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = S_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = S_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
  annotate("point", x = b_cut, y = S_ext[idx_b], color = "black", size = 2.8) +
  geom_line(linewidth = 1.1) +
  facet_wrap(~ Family, ncol = 3) +
  scale_color_manual(values = cols_8) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none", strip.text = element_text(face = "bold", size = 9.5)) +
  labs(
    title = "C. Blended Survival Curves S(t): Gradual Departure vs. Prolonged Trial-Hugging Followed by Rapid Convergence",
    subtitle = "Notice in the Right Panel how S_ble(t) hugs S_obs(t) closely until ~6.5 years before rapidly transitioning to S_ext(t) between 6.5y and 10y",
    x = "Time since randomisation (years)",
    y = "Survival"
  )

p_row4_h <- ggplot(df_all_methods, aes(x = time, y = Hazard, color = Method, linetype = Method)) +
  annotate("rect", xmin = a_cut, xmax = b_cut, ymin = 0, ymax = 0.65, fill = "grey92", alpha = 0.6) +
  geom_vline(xintercept = c(a_cut, b_cut), linetype = "dotted", color = "grey30") +
  geom_line(data = df_ref_curves, aes(x = time, y = h_obs), color = "#7CAE00", linetype = "twodash", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(data = df_ref_curves, aes(x = time, y = h_ext), color = "#00BFC4", linetype = "dashed", linewidth = 0.9, inherit.aes = FALSE) +
  geom_line(linewidth = 1.05) +
  facet_wrap(~ Family, ncol = 3) +
  scale_color_manual(values = cols_8) +
  scale_x_continuous(breaks = c(0, a_cut, 8, b_cut, 16, 20),
                     labels = c("0", "a=4", "8", "b=13", "16", "20")) +
  coord_cartesian(ylim = c(0, 0.65)) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom", legend.direction = "vertical",
        legend.title = element_blank(), strip.text = element_text(face = "bold", size = 9.5)) +
  guides(color = guide_legend(ncol = 2), linetype = guide_legend(ncol = 2)) +
  labs(
    title = "D. Implied Blended Hazard Rates h_ble(t) Across All 8 Entropy, Barycenter & Meta-Analytic Methods",
    subtitle = "Left: boundary jump from linear p=1 slope. Center: smooth monotonic hazard. Right: smooth at boundaries with mid-interval catch-up hazard peak!",
    x = "Time since randomisation (years)",
    y = "Hazard"
  )

fig_peaky <- p_row1_w / p_row2_dw / p_row3_s / p_row4_h + plot_layout(heights = c(1, 0.95, 1, 1.35))
ggsave("docs/fig_entropy_and_peaky_scurves.png", fig_peaky, width = 16, height = 16, dpi = 300)
cat("\nSaved full comparison of original entropy, meta-analytic, and peaky S-curve methods to docs/fig_entropy_and_peaky_scurves.png\n")
