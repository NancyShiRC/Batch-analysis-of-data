library(readxl)
library(PRROC)

# try(Sys.setlocale("LC_ALL", "zh_CN.UTF-8"), silent = TRUE)

base_dir <- "/work_dir"
input_file <- file.path(base_dir, "model_output.xlsx")
out_dir <- file.path(base_dir, "precision_recall_curve")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

plot_style <- list(
  curve_lwd = 3.0,
  axis_lwd = 1.5,
  title_size = 18,
  axis_title_size = 18,
  axis_text_size = 16,
  legend_text_size = 15,
  base_pointsize = 12
)

models <- list(
  list(name = "Clinical", legend = "Clinical", sheet = "clinical", score_col = "Clinical_label_1", color = "#E69F00"),
  list(name = "uniHE", legend = "HE", sheet = "uniHE", score_col = "label_1", color = "#5e65e6"),
  list(name = "uniIHC", legend = "Ki-67", sheet = "uniIHC", score_col = "label_1", color = "#418e8d"),
  list(name = "uniHEIHC", legend = "HE + Ki-67", sheet = "uniHEIHC", score_col = "label_1", color = "#fc0000")
)

cohorts <- list(
  list(name = "Training", title = "Train Cohort", output = "PRROC_PR_train.pdf", model_group = "train", clinical_group = "Train"),
  list(name = "Internal Validation", title = "Internal Validation Cohort", output = "PRROC_PR_internal.pdf", model_group = "internal", clinical_group = "Internal Validation"),
  list(name = "External Validation", title = "External Validation Cohort", output = "PRROC_PR_external.pdf", model_group = "external", clinical_group = "External Validation")
)

read_model_data <- function(model, cohort) {
  dat <- read_excel(input_file, sheet = model$sheet)
  group_value <- if (model$sheet == "clinical") cohort$clinical_group else cohort$model_group
  dat <- dat[dat$group == group_value, ]

  score_col <- model$score_col
  if (!score_col %in% names(dat) && "label_1" %in% names(dat)) {
    score_col <- "label_1"
  }
  if (!score_col %in% names(dat)) {
    stop(paste0("Missing score column for ", model$name, ": ", model$score_col))
  }

  data.frame(
    SampleID = dat$SampleID,
    label = as.integer(dat$true_label),
    score = as.numeric(dat[[score_col]]),
    stringsAsFactors = FALSE
  )
}

calc_prroc_curve <- function(label, score) {
  keep <- !is.na(label) & !is.na(score)
  label <- as.integer(label[keep])
  score <- as.numeric(score[keep])

  positive_scores <- score[label == 1]
  negative_scores <- score[label == 0]
  if (length(positive_scores) == 0 || length(negative_scores) == 0) {
    stop("PRROC requires at least one positive and one negative sample.")
  }

  pr <- PRROC::pr.curve(
    scores.class0 = positive_scores,
    scores.class1 = negative_scores,
    curve = TRUE
  )

  list(
    recall = pr$curve[, 1],
    precision = pr$curve[, 2],
    auprc = pr$auc.integral,
    n = length(label),
    positive_n = length(positive_scores),
    negative_n = length(negative_scores)
  )
}

open_pdf <- function(output_path) {
  if (capabilities("aqua")) {
    grDevices::quartz(
      file = output_path,
      type = "pdf",
      width = 6.2,
      height = 7.2,
      family = "Arial",
      pointsize = plot_style$base_pointsize
    )
  } else {
    grDevices::pdf(
      file = output_path,
      width = 6.2,
      height = 7.2,
      family = "sans",
      pointsize = plot_style$base_pointsize
    )
  }
}

plot_cohort <- function(cohort) {
  curves <- list()
  summary_rows <- list()

  for (model in models) {
    model_dat <- read_model_data(model, cohort)
    pr <- calc_prroc_curve(model_dat$label, model_dat$score)
    curves[[model$name]] <- c(model, pr)
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      cohort = cohort$name,
      model = model$name,
      N = pr$n,
      positive = pr$positive_n,
      negative = pr$negative_n,
      AUPRC = pr$auprc,
      stringsAsFactors = FALSE
    )
  }

  output_path <- file.path(out_dir, cohort$output)
  open_pdf(output_path)
  old_par <- par(no.readonly = TRUE)
  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)

  par(
    mar = c(6, 6, 5, 2),
    mgp = c(3.6, 1, 0),
    tcl = -0.3,
    family = "Arial",
    lwd = plot_style$axis_lwd
  )
  plot(
    NA,
    xlim = c(0, 1),
    ylim = c(0, 1),
    xaxs = "i",
    yaxs = "i",
    main = cohort$title,
    xlab = "Recall",
    ylab = "Precision",
    cex.main = plot_style$title_size / plot_style$base_pointsize,
    cex.lab = plot_style$axis_title_size / plot_style$base_pointsize,
    cex.axis = plot_style$axis_text_size / plot_style$base_pointsize,
    bty = "l"
  )

  for (model in models) {
    curve <- curves[[model$name]]
    lines(curve$recall, curve$precision, col = model$color, lwd = plot_style$curve_lwd, type = "l")
  }

  legend_labels <- vapply(
    models,
    function(model) {
      curve <- curves[[model$name]]
      paste0(model$legend, " AUPRC = ", sprintf("%.3f", curve$auprc))
    },
    character(1)
  )

  legend(
    x = 0.03,
    y = 0.25,
    legend = legend_labels,
    col = vapply(models, function(model) model$color, character(1)),
    lty = 1,
    lwd = plot_style$curve_lwd,
    cex = plot_style$legend_text_size / plot_style$base_pointsize,
    box.lty = 0,
    bg = rgb(1, 1, 1, 0.8),
    x.intersp = 0.8,
    y.intersp = 1.1,
    text.width = 0.42
  )

  do.call(rbind, summary_rows)
}

summary_df <- do.call(rbind, lapply(cohorts, plot_cohort))
summary_path <- file.path(out_dir, "PRROC_PR_curve_summary.csv")
write.csv(summary_df, summary_path, row.names = FALSE)

print(summary_df)
message("Saved PRROC PR curve PDFs to: ", out_dir)
message("Saved summary to: ", summary_path)
