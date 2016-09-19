## try to detect onset

# util --------------------------------------------------------------------------------
rotate <- function(x) t(apply(x, 2, rev))
plot_image <- function(x) image(x, axes = FALSE, col = grey(seq(0, 1, length = 256)), useRaster = T)
mad <- function(x) median(abs(x - median(x))) # median absolute deviation from the median

# init --------------------------------------------------------------------------------
spec_dir <- "data/MLSP 2013/mlsp_contest_dataset/supplemental_data/spectrograms/"
file_names <- list.files(spec_dir)

# notes -------------------------------------------------------------------------------
# columns index rows for some reason
# bottom most row is #1

# load
# good low signal example: 14
ind <- ind + 1
spec <- bmp::read.bmp(paste0(spec_dir, file_names[ind]))
spec <- spec/max(spec) # norm to 1
t <- nrow(spec)
freq <- ncol(spec)

# plot
plot_image(rotate(spec))

## NOISE REDUCTION
# TODO: get better average noise spec
noise_file <- "PC1_20090513_050000_0010.bmp"
noise <- rotate(bmp::read.bmp(paste0(spec_dir, noise_file)))
noise_profile <- apply(noise, 2, mean)/max(apply(noise, 2, mean))

plot(noise_profile, pch = '.')

spec_denoised <-  rotate(spec/rev(noise_profile)) # try subtraction?
plot_image(spec_denoised)

# summary stats
power_sd <- apply(spec, 2, sd)
power_mean <- apply(spec, 2, mean)

power_mad <- apply(spec, 2, mad)
power_median <- apply(spec, 2, median)

all_mean <- mean(spec[, 1:30])
all_sd <- sd(spec[, 1:30])

# find time periods with power anomalies
# mask <- spec > all_mean + all_sd
# plot_image(mask)
mask <- spec > (power_median + power_mad)
plot_image(mask)

n_dev <- apply(mask, 1, sum)

hist(n_dev)
dev_median <- median(n_dev)
dev_mad <- mad(n_dev)

is_song <- n_dev > (dev_median + (dev_mad * 3))
# is_song
ind_plot <- (1:t)[is_song]
plot_image(spec)
points(ind_plot/t, rep(0.2, length(ind_plot)), type = 'p', pch = '-', col = 'red', bg = 'red')

## NEXT:
# hypothesis: noise is symmetric, other stuff is not -- only look for power outliers
# in frequency bands where median != mean. See plot below
# also: low frequency always has lots of power -- try scaling power in low f down by
# some average measure of power in just noise files



## frequency outliers: ---------------
# plot mean vs median, can be used for presence of song in a particular freq band
# not good with short chips, clicks, etc.
#
# hypothesis: noise is symmetric, other stuff is not -- only look for power outliers
# in frequency bands where median != mean.
x_range <- c(min(power_median), max(power_median))
y_range <- c(min(power_mean), max(power_mean))
data.frame('freq' = 1:freq,
           'mean' = power_mean,
           'median' = power_median) %>%
  ggplot() +
  geom_point(aes(x = `median`, y = `mean`,color = `freq`)) +
  geom_line(aes(x = `freq`, y = `freq`)) +
  coord_cartesian(xlim = x_range, ylim = y_range) +
  theme_bw()


#### wtf R
test <- matrix(c(100, 50, 25, 10, 5, 2), nrow = 2)

## Thoughts
# feature ideas:
# - autocorrelation or some other temporal correlation -- need to capture temporal structure
