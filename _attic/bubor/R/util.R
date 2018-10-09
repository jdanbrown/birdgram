#' rotate 90deg
#' @export rotate
rotate <- function(x) t(apply(x, 2, rev))

#' plot_image, with rotate
#' @export plot_image
plot_image <- function(x) image(rotate(x), axes = FALSE, col = grey(seq(0, 1, length = 256)), useRaster = T)

#' mad
#' @export mad
mad <- function(x, na.rm = TRUE) median(abs(x - median(x, na.rm = na.rm)), na.rm = na.rm) # median absolute deviation from the median
