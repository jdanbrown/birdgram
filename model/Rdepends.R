options(
  repos = c(CRAN = "https://cran.rstudio.com/")
)

# lookup
devtools::install_github("jimhester/lookup")

# Dev version of ggplot (e.g. for geom_sf, to be released in 2.3.0)
devtools::install_github("tidyverse/ggplot2")

# viridis
install.packages("viridis")

# Spatial: ggmap, etc.
devtools::install_github("dkahle/ggmap")
install.packages("sf") # >= 0.6-3 (for https://github.com/r-spatial/sf/commit/812cdff)
install.packages("rnaturalearth")
install.packages("rnaturalearthdata") # For ne_coastlines
install.packages("rnaturalearthhires", repos = "http://packages.ropensci.org", type = "source") # For ne_states
install.packages("rgeos") # rnaturalearth: For returnclass="sf"

# seewave
install.packages("seewave")
install.packages("fftw") # TODO Is this necessary in addition to fftw=3.3.7 in envionment.yml?
