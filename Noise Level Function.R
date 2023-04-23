# Basic script for testing Noise levels function


# Load the cell image dataset and preprocess the data
library(keras)
dataset <- dataset_mnist()
x_train <- dataset$train$x
y_train <- dataset$train$y
x_test <- dataset$test$x
y_test <- dataset$test$y
input_shape <- c(28, 28, 1)
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Define a function to add Gaussian noise to the images
add_gaussian_noise <- function(images, mean = 0, sd = 0.1) {
  noisy_images <- array(0, dim = dim(images))
  for (i in 1:nrow(images)) {
    noisy_images[i,,,] <- images[i,,,] + rnorm(1, mean, sd, ncol(images)*nrow(images))
  }
  return(noisy_images)
}

# Train the CNN model on the clean training data
model <- keras_model_sequential()
model %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")
model %>% compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = c("accuracy"))
model %>% fit(x_train, y_train, epochs = 5, batch_size = 32)

# Evaluate the model on the clean test data
clean_scores <- model %>% evaluate(x_test, y_test)
print(paste0("Clean Test Accuracy: ", clean_scores[[2]]))

# Evaluate the model on noisy test data
noise_levels <- c(0, 0.1, 0.2, 0.4)
for (r in noise_levels) {
  noisy_test <- noisy_images(x_test, images = r)
  scores <- model %>% evaluate(noisy_test, y_test)
}
