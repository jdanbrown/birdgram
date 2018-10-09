source(global.R)

## MLSP 2013 labels
dir_mlsp_essential_data <- paste0(dir_bubo, "/data/MLSP 2013/mlsp_contest_dataset/essential_data/")

## ------------------------------------------------------------------------------------
## Format labels

recid2fn <- read.csv(paste0(dir_mlsp_essential_data, "rec_id2filename.txt"), row.names = 1)
labels <- read.csv(paste0(dir_mlsp_essential_data, "rec_labels_test_hidden.csv"), row.names = NULL, header = T, stringsAsFactors = F)

# Format table
labels_f <- labels %>%
  mutate(fold = ifelse(label_1 == "?", "test", "train"),
         label_1_orig = label_1,
         label_1 = as.numeric(ifelse(label_1 == "?", "", label_1)))

# filnamea of clips containing thrush
recid2fn[labels_f[labels_f$label_1 == 9 & !is.na(labels_f$label_1), "rec_id"], "filename"]
