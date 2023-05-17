library(EBImage)
library(tidyverse)
library(pracma)
library(randomForest)
library(ggimage)
library(keras)
input_shape = c(64, 64, 1)
lr = 0.001
model_function <- function(learning_rate = lr) {
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = input_shape) %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_dropout(rate = 0.25) %>% 
    layer_flatten() %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = 64) %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = 28, activation = 'softmax')
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = "accuracy"
  )
  
  return(model)
  
}

model_function_rmsprop <- function(learning_rate = lr) {
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = input_shape) %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_dropout(rate = 0.25) %>% 
    layer_flatten() %>% 
    layer_dense(units = 128, activation = 'relu') %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = 64) %>% 
    layer_dropout(rate = 0.5) %>% 
    layer_dense(units = 28, activation = 'softmax')
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(learning_rate = learning_rate),
    metrics = "accuracy"
  )
  
  return(model)
  
}

model_original <- model_function()
model_rmsprop_original <- model_function_rmsprop()
model <- model_function()
model_rmsprop <- model_function_rmsprop()

load_model_weights_tf(model, "Image_App/models/cnn_catent_cweights_comb_weights/cnn_catent_cweights_comb_weights")
load_model_weights_tf(model_original, "Image_App/models/cnn_catent_cweights_weights/cnn_catent_cweights_weights")
load_model_weights_tf(model_rmsprop, "Image_App/models/cnn_rmsprop_cweights_comb/weights")
load_model_weights_tf(model_rmsprop_original, "Image_App/models/cnn_rmsprop_cweights_weights/cnn_rmsprop_cweights_weights")

labels = list.files("../bio_data/Biotechnology/data_processed/cell_images/")
cell_boundaries_raw = read.csv("../bio_data/Biotechnology/data_processed/cell_boundaries.csv.gz")

cluster_ids = list()

for (i in 1:length(labels)) {
  
  cluster_file = list.files(paste0("../bio_data/Biotechnology/data_processed/cell_images/", labels[i]))
  
  cluster_ids[i] <- list(gsub(".*cell_|.png", "", cluster_file))
}

cluster_imgs = list()

for (i in 1:length(labels)) {
  
  cluster_file = list.files(paste0("../bio_data/Biotechnology/data_processed/cell_images/", labels[i]), full.names=T)
  
  cluster_imgs[i] <- list(sapply(cluster_file, readImage, simplify = F))
  
}

get_inside = function(cellID, img, cell_boundaries) {
  
  cell_boundary = cell_boundaries |>
    filter(cell_id %in% cellID)
  
  # rescale the boundary according to the pixels
  pixels = dim(img)
  cell_boundary$vertex_x_scaled <- 1+((cell_boundary$vertex_x - min(cell_boundary$vertex_x))/0.2125)
  cell_boundary$vertex_y_scaled <- 1+((cell_boundary$vertex_y - min(cell_boundary$vertex_y))/0.2125)
  
  # identify which pixels are inside or outside of the cell segment using inpolygon
  pixel_locations = expand.grid(seq_len(nrow(img)), seq_len(ncol(img)))
  
  pixels_inside = inpolygon(x = pixel_locations[,1],
                            y = pixel_locations[,2],
                            xp = cell_boundary$vertex_x_scaled,
                            yp = cell_boundary$vertex_y_scaled,
                            boundary = TRUE)
  
  img_inside = img
  img_inside@.Data <- matrix(pixels_inside, nrow = nrow(img), ncol = ncol(img))
  
  return(img_inside)
}

mask_resize = function(img, img_inside, w = 50, h = 50) {
  
  img_mask = img*img_inside
  
  # then, transform the masked image to the same number of pixels, 50x50
  img_mask_resized = resize(img_mask, w, h)
  
  return(img_mask_resized)
}

cluster_imgs_masked_resized = list()

for (i in 1:length(labels)) {
  cell_boundaries = cell_boundaries_raw |>
    filter(cell_id %in% cluster_ids[[i]])
  
  cluster_imgs_inside <- mapply(get_inside, cluster_ids[[i]], cluster_imgs[[i]], MoreArgs = list(cell_boundaries = cell_boundaries), SIMPLIFY = FALSE)
  
  cluster_imgs_masked_resized[i] = list(mapply(mask_resize, cluster_imgs[[i]], cluster_imgs_inside, MoreArgs = list(w = 64, h = 64), SIMPLIFY = FALSE))
  
}

imgs_masked_resize_64 = do.call(c, cluster_imgs_masked_resized)

num_images = length(imgs_masked_resize_64)
img_names = names(imgs_masked_resize_64)

x = array(dim = c(num_images, 64, 64, 1))

# Centering Intensity values
for (i in 1:num_images) {
  x[i,,,1] <- imgs_masked_resize_64[[i]]@.Data - mean(imgs_masked_resize_64[[i]]@.Data)
}

input_shape = dim(x)[2:4]

times = c()

for (i in 1:length(labels)) {
  times[i] = length(cluster_ids[[i]])
}

y = factor(rep(labels, times = times))

yy = model.matrix(~ y - 1)

# Shuffle data into training and test set

shuf_ind <- sample(1:dim(x)[1], dim(x)[1]*0.8)
shuf_x_train <- x[shuf_ind, , ,]
shuf_x_train <- array(shuf_x_train, dim = c(dim(shuf_x_train)[1], dim(shuf_x_train)[2], dim(shuf_x_train)[3], 1))

shuf_x_test <- x[-shuf_ind, , ,]
shuf_x_test <- array(shuf_x_test, dim = c(dim(shuf_x_test)[1], dim(shuf_x_test)[2], dim(shuf_x_test)[3], 1))

shuf_yy_train <- yy[shuf_ind, ]
shuf_yy_test <- yy[-shuf_ind, ]

pred <- model |> predict(shuf_x_test)
pred_class <- apply(pred, 1, which.max)

original_pred <- model_original |> predict(shuf_x_test)
original_pred_class <- apply(original_pred, 1, which.max)

rms_pred <- model_rmsprop |> predict(shuf_x_test)
rms_pred_class <- apply(rms_pred, 1, which.max)

# library(mrtree)
plotContTable <- function(est_label, true_label, true_label_order = NULL, est_label_order = NULL,
                          short.names = NULL, xlab = "True Labels", ylab = "Predicted Labels") {
  
  requireNamespace("ggplot2")
  if (!is.null(true_label_order)) {
    checkmate::assert_true(all(sort(unique(true_label)) == sort(true_label_order)))
    true_label = factor(true_label, levels = true_label_order)
  }
  if (!is.null(est_label_order)) {
    checkmate::assert_true(all(sort(unique(est_label)) == sort(est_label_order)))
    # est_label = factor(est_label, levels=est_label_order)
  }
  if (is.null(short.names)) {
    short.names = levels(factor(true_label))
    
  }
  cont.table <- table(true_label, est_label)
  if (!is.null(true_label_order)) {
    cont.table = cont.table[true_label_order, ]
  }
  if (!is.null(est_label_order)) {
    cont.table = cont.table[, est_label_order]
  }
  K <- ncol(cont.table)
  sub.clusters <- paste0("cluster ", colnames(cont.table))
  cont.table <- apply(as.matrix(cont.table), 2, as.integer)
  cont.table <- data.frame(cont.table)
  cont.table$Reference = factor(short.names, levels = short.names)
  colnames(cont.table) <- c(sub.clusters, "Reference")
  dat3 <- reshape2::melt(cont.table, id.var = "Reference")
  grid.labels = as.character(dat3$value)
  grid.labels[grid.labels == "0"] = ""
  g <- ggplot(dat3, aes(x = Reference, y = variable)) + geom_tile(aes(fill = value)) +
    geom_text(aes(label = grid.labels), size = 4.5) + scale_fill_gradient(low = "white",
                                                                          high = "purple") + labs(y = ylab, x = xlab) + theme(panel.background = element_blank(),
                                                                                                                              axis.line = element_blank(), axis.text.x = element_text(size = 13, angle = 90),
                                                                                                                              axis.text.y = element_text(size = 13), axis.ticks = element_blank(), axis.title.x = element_text(size = 18),
                                                                                                                              axis.title.y = element_text(size = 18), legend.position = "none")
  return(g)
}

order = c("cluster_1", "cluster_10", "cluster_11", "cluster_12", "cluster_13",
          "cluster_14", "cluster_15", "cluster_16", "cluster_17", "cluster_18",
          "cluster_19", "cluster_2", "cluster_20", "cluster_21", "cluster_22",
          "cluster_23", "cluster_24", "cluster_25", "cluster_26", "cluster_27",
          "cluster_28", "cluster_3", "cluster_4", "cluster_5", "cluster_6",
          "cluster_7", "cluster_8", "cluster_9")

plotContTable(pred_class, y[-shuf_ind])
plotContTable(original_pred_class, y[-shuf_ind])
plotContTable(rms_pred_class, y[-shuf_ind])

1-(sum(diag(table(original_pred_class, y[-shuf_ind])))/length(y[-shuf_ind]))


