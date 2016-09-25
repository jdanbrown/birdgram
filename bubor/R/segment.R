#' find putative song epochs
#'
#' How it works:
#' \enumerate{
#'   \item Find pixels that exceed some multiple of the absolute median deviation for each frequency. Specified by \code{freq_mad_threshold}.
#'   \item Smooth the time series of power deviations (window_length).
#'   \item Find time points when power deviates for > threshold number/proportion of frequencies.
#' }
#'
#' @param spec [matrix] Spectrogram bitmap, normalized to 1. Expected to have dim = (freq
#' x time).
#' @param plots, verbose [logical] flags for whether to show debugging plots and text
#' @param freq_mad_threshold frequency deviation threshold, in units of standard absolute median devs.
#' @param window_length moving average window length
#' @param threshold [0/1]
#'
#' @export segment
segment <- function(spec, plots = FALSE, verbose = FALSE, freq_mad_threshold = 2,
                    window_length = 3
                    # , threshold = 5
                    ) {

  # Outlier mask using median and median absolute deviation (mad) -----------------------
  # for [f*t] matrix, get median value at each frequency, then each time point

  median_power <- list("freq" = apply(spec, 1, median), "time" = apply(spec, 2, median))
  mad_power <- list("freq" = apply(spec, 1, mad), "time"  = apply(spec, 2, mad))

  mask <- (
    spec > (median_power[["freq"]] + mad_power[["freq"]] * freq_mad_threshold)  #freq deviates
    # & spec > (median_power[["time"]] + mad_power[["time"]] * 2) #time deviates
  )

  # if (plots) {
  #   plot_image(mask)
  # }

  # smooth
  window <- rep(1/window_length, window_length)
  norm_dev <- apply(mask, 2, sum) / max(apply(mask, 2, sum)) # total deviations at each time point #in each frequency band
  norm_dev_smoothed <- filter(norm_dev, window, sides = 2)
  norm_dev_smoothed <- norm_dev_smoothed
  # n_dev_norm <- n_dev_smoothed/max(n_dev_smoothed, na.rm = TRUE)

  median_deviations <- median(norm_dev_smoothed, na.rm = TRUE)
  mean_deviations <- mean(norm_dev_smoothed, na.rm = TRUE)
  mad_deviations <- mad(norm_dev_smoothed, na.rm = TRUE)


  scaling_ratio <- min(c((median_deviations/mean_deviations), 1)) ^ 20
  # scaling_ratio <- 1
  threshold <- median_deviations + (mad_deviations * 1 * scaling_ratio)

   if (plots) {

     # par(mfrow = c(2, 1))
     # plot_image(mask)
    hist(norm_dev_smoothed)

    t <- dim(spec)[2]
    plot(norm_dev, type = 'l')
    lines(norm_dev_smoothed, col = 'red')
    lines(1:t, rep(median_deviations + (mad_deviations * 6), t), col = "cyan")
    lines(1:t, rep(threshold, t), col = "blue")
    lines(1:t, rep(median_deviations, t), col = "black")
    # lines(1:t, rep(mean_deviations, t), col = "cyan")

    # lines(1:t, rep(median_deviations + (mad_deviations * param[2]), t), col = "blue")
  }


  # if (plots) {
  #   hist(norm_dev_smoothed,
  #        xlab = "deviations per frequency", ylab = "n",
  #        main = sprintf("Number of Deviations\nmedian = %2.f, MAD = %2.f", median_deviations, mad_deviations))
  # }

  # is_song <- n_dev > (median_deviations + (mad_deviations * param[2]))



  # is_song <- norm_dev_smoothed > (median_deviations + (mad_deviations * param[2]))
  print(sprintf("median = %2.2f, mad = %2.2f", median_deviations, mad_deviations))
  print(sprintf("med/mean ratio = %2.2f", median_deviations/mean_deviations))
  print(sprintf("scaling ratio = %2.2f", scaling_ratio))
  print(sprintf("threshold = %2.2f", threshold))
  is_song <- norm_dev_smoothed > threshold

  if (plots) {
    ind_plot <- (1:dim(spec)[2])[is_song]
    plot_image(spec)
    points(ind_plot/dim(spec)[2], rep(0.2, length(ind_plot)), type = 'p', pch = '-', col = 'red', bg = 'red')
  }

  return(is_song)
}