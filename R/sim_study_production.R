# ==============================================================================
# Dynamic Survival Blending via Information Geometry & Relative Uncertainty
# Production Simulation Study Pipeline (ADEMP Protocol)
# Target Journal: Medical Decision Making / Value in Health
# ==============================================================================

# Required libraries
required_pkgs <- c("survival", "flexsurv", "dplyr", "tidyr", "ggplot2", "patchwork")
new_pkgs <- required_pkgs[!required_pkgs %in% installed.packages()[, "Package"]]
if (length(new_pkgs) > 0) install.packages(new_pkgs)

library(survival)
library(flexsurv)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)

# ------------------------------------------------------------------------------
# 1. Performance Metric Functions with MCSE (Morris et al., 2019)
# ------------------------------------------------------------------------------
calc_bias <- function(theta_hat, theta_true) {
  valid <- theta_hat[!is.na(theta_hat)]
  n_val <- length(valid)
  est_bias <- mean(valid) - theta_true
  mcse <- sqrt(sum((valid - mean(valid))^2) / (n_val * (n_val - 1)))
  c(Est = est_bias, MCSE = mcse)
}

calc_emp_se <- function(theta_hat) {
  valid <- theta_hat[!is.na(theta_hat)]
  n_val <- length(valid)
  est_se <- sqrt(sum((valid - mean(valid))^2) / (n_val - 1))
  mcse <- est_se / sqrt(2 * (n_val - 1))
  c(Est = est_se, MCSE = mcse)
}

calc_rmse <- function(theta_hat, theta_true) {
  valid <- theta_hat[!is.na(theta_hat)]
  n_val <- length(valid)
  mse <- mean((valid - theta_true)^2)
  est_rmse <- sqrt(mse)
  mcse <- sqrt(sum(((valid - theta_true)^2 - mse)^2) / (4 * mse * n_val * (n_val - 1)))
  c(Est = est_rmse, MCSE = mcse)
}

calc_coverage <- function(theta_true, theta_low, theta_upp) {
  valid <- !is.na(theta_low) & !is.na(theta_upp)
  n_sim <- sum(valid)
  covered <- ifelse(theta_low[valid] <= theta_true & theta_upp[valid] >= theta_true, 1, 0)
  est_cov <- sum(covered) / n_sim
  mcse <- sqrt((est_cov * (1 - est_cov)) / n_sim)
  c(Est = est_cov, MCSE = mcse)
}

trapezoidal_area <- function(y, diff_step = 2) {
  n <- length(y)
  0.5 * diff_step * (y[1] + y[n] + 2 * sum(y[2:(n - 1)]))
}

get_median_survival <- function(S_vec, t_vec) {
  idx <- which(S_vec <= 0.50)
  if (length(idx) == 0) return(max(t_vec))
  min(t_vec[idx])
}

# ------------------------------------------------------------------------------
# 2. Geometric Blending Engine (Che et al. 2023 / Log-Linear Barycenter)
# ------------------------------------------------------------------------------
calc_geometric_blend <- function(S_obs, S_ext, h_obs, h_ext, pi_vec, t_vec) {
  dt_step <- diff(t_vec)[1]
  S_ble <- (S_obs^(1 - pi_vec)) * (S_ext^pi_vec)
  
  # Numerical derivative pi'(t)
  dpi <- c(diff(pi_vec) / dt_step, tail(diff(pi_vec) / dt_step, 1))
  
  # Analytical blended hazard: h_ble = (1-pi)*h_obs + pi*h_ext + pi' * ln(S_obs/S_ext)
  log_ratio <- log(pmax(S_obs, 1e-8) / pmax(S_ext, 1e-8))
  h_ble <- (1 - pi_vec) * h_obs + pi_vec * h_ext + dpi * log_ratio
  h_ble <- pmax(h_ble, 1e-8)
  
  # Blended event density
  f_ble <- S_ble * h_ble
  list(S = S_ble, h = h_ble, f = f_ble)
}

# Dynamic Restricted Kullback-Leibler (DRKL) divergence
calc_drkl <- function(f_true, f_ble, t_vec, t1, t2) {
  idx <- which(t_vec >= t1 & t_vec <= t2)
  dt_step <- diff(t_vec)[1]
  integrand <- f_true[idx] * log(pmax(f_true[idx], 1e-8) / pmax(f_ble[idx], 1e-8))
  sum(integrand) * dt_step
}

# ------------------------------------------------------------------------------
# 3. Ground Truth Data-Generating Mechanism (SCLC Clinical Trial)
# ------------------------------------------------------------------------------
t_grid <- seq(1, 750, by = 2)
dt     <- diff(t_grid)[1]

# True SCLC Patient Survival: Weibull(shape = 1.15, scale = 380 days)
S_true <- pweibull(t_grid, shape = 1.15, scale = 380, lower.tail = FALSE)
h_true <- (1.15 / 380) * (t_grid / 380)^(1.15 - 1)
f_true <- dweibull(t_grid, shape = 1.15, scale = 380)

true_rmst_750 <- trapezoidal_area(S_true, diff_step = dt)
true_t50      <- get_median_survival(S_true, t_grid)

# External Registry Anchor: Weibull(shape = 1.20, scale = 365 days)
S_ext  <- pweibull(t_grid, shape = 1.20, scale = 365, lower.tail = FALSE)
h_ext  <- (1.20 / 365) * (t_grid / 365)^(1.20 - 1)
se_ext <- rep(0.015, length(t_grid)) # Large mature registry with stable SE

cat(sprintf("Ground Truth Parameters (750-Day Horizon):\n  RMST: %.2f days\n  Median T50: %.2f days\n\n",
            true_rmst_750, true_t50))

# ------------------------------------------------------------------------------
# 4. Core Factorial Simulation Loop
# ------------------------------------------------------------------------------
run_dynblend_simulation <- function(n_reps = 100, seed = 2026) {
  set.seed(seed)
  
  factorial_grid <- expand.grid(
    N_arm = c(75, 250),
    ACT   = c(100, 360),
    BL    = c(180, 360),
    stringsAsFactors = FALSE
  )
  
  methods <- c(
    "Unadjusted", "Piecewise",
    "Beta(1,1)", "Beta(1,5)", "Beta(3,3)", "Beta(5,1)",
    "KL_Clock", "JSD_Clock", "Hellinger_Clock", "Wasserstein_Clock",
    "Precision_Barycenter"
  )
  
  results_list <- list()
  zipper_data  <- list()
  
  for (scen in seq_len(nrow(factorial_grid))) {
    n_p <- factorial_grid$N_arm[scen]
    act <- factorial_grid$ACT[scen]
    bl  <- factorial_grid$BL[scen]
    t1  <- act
    t2  <- min(act + bl, 748)
    
    scen_label <- sprintf("N=%d, ACT=%dd, BL=%dd", n_p, act, bl)
    cat(sprintf("Running Scenario %d/%d: %s\n", scen, nrow(factorial_grid), scen_label))
    
    # Preallocate metrics
    rmst_mat <- matrix(NA, nrow = n_reps, ncol = length(methods), dimnames = list(NULL, methods))
    low_mat  <- matrix(NA, nrow = n_reps, ncol = length(methods), dimnames = list(NULL, methods))
    upp_mat  <- matrix(NA, nrow = n_reps, ncol = length(methods), dimnames = list(NULL, methods))
    t50_mat  <- matrix(NA, nrow = n_reps, ncol = length(methods), dimnames = list(NULL, methods))
    drkl_mat <- matrix(NA, nrow = n_reps, ncol = length(methods), dimnames = list(NULL, methods))
    
    idx_blend <- which(t_grid >= t1 & t_grid <= t2)
    taper     <- sin(pi * (t_grid[idx_blend] - t1) / (t2 - t1))^2
    u_norm    <- pmin(pmax((t_grid - t1) / (t2 - t1), 0), 1)
    
    for (rep in seq_len(n_reps)) {
      # 1. Simulate internal trial data (immature plateau with administrative censoring)
      t_raw <- rweibull(n_p, shape = 0.95, scale = 440)
      c_rnd <- runif(n_p, 0, act)
      s_ind <- rbinom(n_p, 1, 0.85)
      t_adm <- pmin(t_raw, act)
      t_obs <- ifelse(s_ind == 1, t_adm, pmin(t_adm, c_rnd))
      d_obs <- as.numeric(t_raw == t_obs)
      
      df_trial <- data.frame(time = t_obs, status = d_obs)
      
      # 2. Fit parametric trial model
      fit_o <- tryCatch(
        flexsurvreg(Surv(time, status) ~ 1, data = df_trial, dist = "weibull"),
        error = function(e) flexsurvreg(Surv(time, status) ~ 1, data = df_trial, dist = "exp")
      )
      
      pred_o <- summary(fit_o, t = t_grid, type = "survival", se = TRUE)[[1]]
      S_obs  <- pred_o$est
      se_obs <- pmax(pred_o$se, 1e-4)
      S_low  <- pmax(pred_o$lcl, 1e-6)
      S_upp  <- pmin(pred_o$ucl, 1.0)
      
      pred_h <- summary(fit_o, t = t_grid, type = "hazard")[[1]]
      h_obs  <- pmax(pred_h$est, 1e-6)
      
      # 3. Compute Weight Schedules for the Candidate Estimators
      w_schedules <- list()
      
      # Static Betas
      w_schedules[["Beta(1,1)"]] <- pbeta(u_norm, 1, 1)
      w_schedules[["Beta(1,5)"]] <- pbeta(u_norm, 1, 5)
      w_schedules[["Beta(3,3)"]] <- pbeta(u_norm, 3, 3)
      w_schedules[["Beta(5,1)"]] <- pbeta(u_norm, 5, 1)
      
      # Approach 1a: KL Divergence Clock
      kl_pt <- 0.5 * (log(se_ext^2 / se_obs^2) + (se_obs^2 + (S_obs - S_ext)^2) / se_ext^2 - 1)
      kl_pt <- pmax(kl_pt, 1e-6)
      cum_kl <- cumsum(kl_pt[idx_blend] * taper) * dt
      pi_kl <- ifelse(t_grid < t1, 0, ifelse(t_grid > t2, 1, 0))
      pi_kl[idx_blend] <- cum_kl / max(cum_kl)
      w_schedules[["KL_Clock"]] <- pi_kl
      
      # Approach 1b: Jensen-Shannon Divergence (JSD) Clock
      var_m <- 0.5 * (se_obs^2 + se_ext^2)
      mu_m  <- 0.5 * (S_obs + S_ext)
      kl_pm <- 0.5 * (log(var_m / se_obs^2) + (se_obs^2 + (S_obs - mu_m)^2) / var_m - 1)
      kl_qm <- 0.5 * (log(var_m / se_ext^2) + (se_ext^2 + (S_ext - mu_m)^2) / var_m - 1)
      jsd_pt <- pmax(0.5 * kl_pm + 0.5 * kl_qm, 1e-6)
      cum_jsd <- cumsum(jsd_pt[idx_blend] * taper) * dt
      pi_jsd <- ifelse(t_grid < t1, 0, ifelse(t_grid > t2, 1, 0))
      pi_jsd[idx_blend] <- cum_jsd / max(cum_jsd)
      w_schedules[["JSD_Clock"]] <- pi_jsd
      
      # Approach 1c: Hellinger Distance Clock
      bc <- sqrt(2 * se_obs * se_ext / (se_obs^2 + se_ext^2)) *
        exp(-0.25 * (S_obs - S_ext)^2 / (se_obs^2 + se_ext^2))
      hel_pt <- pmax(1 - bc, 1e-6)
      cum_hel <- cumsum(hel_pt[idx_blend] * taper) * dt
      pi_hel <- ifelse(t_grid < t1, 0, ifelse(t_grid > t2, 1, 0))
      pi_hel[idx_blend] <- cum_hel / max(cum_hel)
      w_schedules[["Hellinger_Clock"]] <- pi_hel
      
      # Approach 1d: 1-Wasserstein Optimal Transport Clock
      wass_pt <- abs(S_obs - S_ext) + abs(se_obs - se_ext)
      cum_wass <- cumsum(wass_pt[idx_blend] * taper) * dt
      pi_wass <- ifelse(t_grid < t1, 0, ifelse(t_grid > t2, 1, 0))
      pi_wass[idx_blend] <- cum_wass / max(cum_wass)
      w_schedules[["Wasserstein_Clock"]] <- pi_wass
      
      # Approach 2: Precision Barycenter (Reverse-KL)
      u_sm <- 3 * u_norm^2 - 2 * u_norm^3
      prec_obs <- 1 / se_obs^2
      prec_ext <- 1 / se_ext^2
      pi_bary  <- (u_sm * prec_ext) / ((1 - u_sm) * prec_obs + u_sm * prec_ext)
      w_schedules[["Precision_Barycenter"]] <- pi_bary
      
      # 4. Evaluate Each Estimator
      for (m in methods) {
        if (m == "Unadjusted") {
          S_m <- S_obs
          f_m <- S_obs * h_obs
          S_m_low <- S_low
          S_m_upp <- S_upp
        } else if (m == "Piecewise") {
          h_pw <- ifelse(t_grid <= t1, h_obs, h_ext)
          S_m  <- exp(-cumsum(h_pw) * dt)
          f_m  <- S_m * h_pw
          S_m_low <- pmax(S_m - 1.96 * se_obs, 1e-6)
          S_m_upp <- pmin(S_m + 1.96 * se_obs, 1.0)
        } else {
          w_vec <- w_schedules[[m]]
          ble_res <- calc_geometric_blend(S_obs, S_ext, h_obs, h_ext, w_vec, t_grid)
          S_m <- ble_res$S
          f_m <- ble_res$f
          
          # Delta-method confidence bounds
          ble_low <- calc_geometric_blend(S_low, pmax(S_ext - 1.96 * se_ext, 1e-6), h_obs, h_ext, w_vec, t_grid)
          ble_upp <- calc_geometric_blend(S_upp, pmin(S_ext + 1.96 * se_ext, 1.0), h_obs, h_ext, w_vec, t_grid)
          S_m_low <- ble_low$S
          S_m_upp <- ble_upp$S
        }
        
        rmst_mat[rep, m] <- trapezoidal_area(S_m, diff_step = dt)
        low_mat[rep, m]  <- trapezoidal_area(S_m_low, diff_step = dt)
        upp_mat[rep, m]  <- trapezoidal_area(S_m_upp, diff_step = dt)
        t50_mat[rep, m]  <- get_median_survival(S_m, t_grid)
        drkl_mat[rep, m] <- calc_drkl(f_true, f_m, t_grid, t1, t2)
      }
      
      # Store zipper data for high-uncertainty scenario (N=75, ACT=100, BL=180)
      if (scen == 1 && rep <= 100) {
        for (m in c("Beta(5,1)", "Beta(3,3)", "Beta(1,5)", "KL_Clock", "Precision_Barycenter")) {
          zipper_data[[paste0(m, "_", rep)]] <- data.frame(
            Method = m, Rep = rep,
            Est = rmst_mat[rep, m], Low = low_mat[rep, m], Upp = upp_mat[rep, m]
          )
        }
      }
    }
    
    # Summarize Performance Metrics for Scenario
    for (m in methods) {
      b_rmst  <- calc_bias(rmst_mat[, m], true_rmst_750)
      se_rmst <- calc_emp_se(rmst_mat[, m])
      rmse_r  <- calc_rmse(rmst_mat[, m], true_rmst_750)
      cov_r   <- calc_coverage(true_rmst_750, low_mat[, m], upp_mat[, m])
      
      b_t50   <- calc_bias(t50_mat[, m], true_t50)
      
      results_list[[paste0(scen, "_", m)]] <- data.frame(
        Scenario_ID  = scen,
        N_arm        = n_p,
        ACT          = act,
        BL           = bl,
        Method       = m,
        RMST_Bias    = b_rmst["Est"],
        RMST_Bias_SE = b_rmst["MCSE"],
        RMST_EmpSE   = se_rmst["Est"],
        RMST_RMSE    = rmse_r["Est"],
        Coverage_95  = cov_r["Est"],
        Cov_MCSE     = cov_r["MCSE"],
        T50_Bias     = b_t50["Est"],
        Mean_DRKL    = mean(drkl_mat[, m], na.rm = TRUE)
      )
    }
  }
  
  df_summary <- bind_rows(results_list)
  df_zipper  <- bind_rows(zipper_data)
  
  list(summary = df_summary, zipper = df_zipper)
}

# ------------------------------------------------------------------------------
# 5. Methodological Demonstration Dashboard Plot
# ------------------------------------------------------------------------------
create_methodology_dashboard <- function(t1 = 100, t2 = 360) {
  idx_b <- which(t_grid >= t1 & t_grid <= t2)
  taper <- sin(pi * (t_grid[idx_b] - t1) / (t2 - t1))^2
  u_n   <- pmin(pmax((t_grid - t1) / (t2 - t1), 0), 1)
  
  # Demo curves
  S_obs_demo  <- exp(-(t_grid / 440)^0.95)
  se_obs_demo <- 0.02 + 0.18 * (t_grid / 750)^2 # Rapidly exploding trial tail
  h_obs_demo  <- (0.95 / 440) * (t_grid / 440)^(0.95 - 1)
  
  # Clocks
  kl_pt <- 0.5 * (log(se_ext^2 / se_obs_demo^2) + (se_obs_demo^2 + (S_obs_demo - S_ext)^2) / se_ext^2 - 1)
  cum_kl <- cumsum(pmax(kl_pt[idx_b], 1e-6) * taper) * dt
  pi_kl <- ifelse(t_grid < t1, 0, ifelse(t_grid > t2, 1, 0))
  pi_kl[idx_b] <- cum_kl / max(cum_kl)
  
  prec_o <- 1 / se_obs_demo^2
  prec_e <- 1 / se_ext^2
  u_sm   <- 3 * u_n^2 - 2 * u_n^3
  pi_bar <- (u_sm * prec_e) / ((1 - u_sm) * prec_o + u_sm * prec_e)
  
  pi_b15 <- pbeta(u_n, 1, 5)
  pi_b33 <- pbeta(u_n, 3, 3)
  pi_b51 <- pbeta(u_n, 5, 1)
  
  # Blended survival & hazards
  ble_kl  <- calc_geometric_blend(S_obs_demo, S_ext, h_obs_demo, h_ext, pi_kl, t_grid)
  ble_bar <- calc_geometric_blend(S_obs_demo, S_ext, h_obs_demo, h_ext, pi_bar, t_grid)
  ble_b33 <- calc_geometric_blend(S_obs_demo, S_ext, h_obs_demo, h_ext, pi_b33, t_grid)
  
  # Panel A: Schedules
  df_w <- data.frame(
    Time = t_grid,
    `Beta(1,5)`   = pi_b15,
    `Beta(3,3)`   = pi_b33,
    `Beta(5,1)`   = pi_b51,
    `KL Clock`    = pi_kl,
    `Barycenter`  = pi_bar,
    check.names   = FALSE
  ) %>%
    filter(Time >= t1 - 20 & Time <= t2 + 20) %>%
    pivot_longer(-Time, names_to = "Schedule", values_to = "Weight")
  
  p1 <- ggplot(df_w, aes(x = Time, y = Weight, color = Schedule, linetype = Schedule)) +
    geom_vline(xintercept = c(t1, t2), linetype = "dashed", color = "grey50") +
    geom_line(linewidth = 1) +
    scale_color_brewer(palette = "Set1") +
    theme_minimal(base_size = 11) +
    labs(title = "A. Dynamic vs. Static Weight Schedules π(t)",
         x = "Time (Days)", y = "External Anchor Weight") +
    theme(legend.position = "bottom")
  
  # Panel B: Survival Extrapolations
  df_s <- data.frame(
    Time = rep(t_grid, 5),
    Survival = c(S_obs_demo, S_ext, ble_b33$S, ble_kl$S, ble_bar$S),
    Model = rep(c("Unadjusted Trial", "External Anchor", "Static Beta(3,3)",
                  "KL Clock", "Precision Barycenter"), each = length(t_grid))
  )
  
  p2 <- ggplot(df_s, aes(x = Time, y = Survival, color = Model, linetype = Model)) +
    geom_rect(aes(xmin = t1, xmax = t2, ymin = 0, ymax = 1),
              fill = "grey92", alpha = 0.02, color = NA, inherit.aes = FALSE) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = c("Unadjusted Trial" = "red3", "External Anchor" = "navy",
                                  "Static Beta(3,3)" = "purple", "KL Clock" = "forestgreen",
                                  "Precision Barycenter" = "darkorange")) +
    theme_minimal(base_size = 11) +
    labs(title = "B. Extrapolated Survival Curves S(t)",
         x = "Time (Days)", y = "Overall Survival S(t)") +
    theme(legend.position = "none")
  
  # Panel C: Implied Hazards
  df_h <- data.frame(
    Time = rep(t_grid, 5),
    Hazard = c(h_obs_demo, h_ext, ble_b33$h, ble_kl$h, ble_bar$h),
    Model = rep(c("Unadjusted Trial", "External Anchor", "Static Beta(3,3)",
                  "KL Clock", "Precision Barycenter"), each = length(t_grid))
  )
  
  p3 <- ggplot(df_h, aes(x = Time, y = Hazard, color = Model, linetype = Model)) +
    geom_rect(aes(xmin = t1, xmax = t2, ymin = 0, ymax = 0.01),
              fill = "grey92", alpha = 0.02, color = NA, inherit.aes = FALSE) +
    geom_line(linewidth = 1) +
    coord_cartesian(ylim = c(0, 0.006)) +
    scale_color_manual(values = c("Unadjusted Trial" = "red3", "External Anchor" = "navy",
                                  "Static Beta(3,3)" = "purple", "KL Clock" = "forestgreen",
                                  "Precision Barycenter" = "darkorange")) +
    theme_minimal(base_size = 11) +
    labs(title = "C. Implied Instantaneous Hazard Rates h(t)",
         subtitle = "Taper guarantees continuous C^0 hazard without boundary kinks",
         x = "Time (Days)", y = "Hazard Rate") +
    theme(legend.position = "none")
  
  p1 / (p2 | p3)
}

cat("Pipeline successfully loaded.\nCall `res <- run_dynblend_simulation(n_reps = 100)` to execute.\n")
