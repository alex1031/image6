# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinydashboard)
library(rdrop2)
library(tidyverse)
library(keras)
library(EBImage)
library(plotly)

# Load the model and data
val_loss <- read.csv("val_loss.csv")

input_shape = c(64, 64, 1)
lr = 0.001
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

model <- model_function_rmsprop()
load_model_weights_tf(model, "models/cnn_rmsprop_cweights_comb")

ui <- fluidPage(
  
  navbarPage(
    "CCR",
    
    tags$head(
      tags$style(HTML("
      /* Set the width and height of the app */
      .shiny-output {
        width: 1000px;
        height: 800px;
      }
      
      .navbar-default .navbar-nav > .active > a, .navbar-default .navbar-nav > .active > a:hover, .navbar-default .navbar-nav > .active > a:focus{
        background-color: #E2DED0;
        font-weight: bolder;
      }
    "))
    ),
    
    
    tabPanel("Introduction",
             mainPanel(
               align = "center",
               h2("Welcome to the Cell Classification Robustness (CCR) Shiny Application"),
               br(),
               p("This interactive tool is designed to help you understand how different factors, such as Gaussian noise and image resolution, can impact the performance of a CNN model."),
               hr()),
             mainPanel(
               align = "center",
               h3("Here's how to navigate and use the application:"),
               br(),
               h4("Introduction Tab: "),
               p("You are currently on the introduction tab, which provides an overview of how to use this application."),
               br(),
               h4("Gaussian Noise Levels Tab: "),
               p("On this tab, you'll see how the model's performance is affected by the introduction of Gaussian noise."),
               br(),
               h4("Image Resolution Tab: "),
               p("This tab helps you understand how image resolution affects the model's performance. It provides a similar layout as the Gaussian Noise Levels Tab, but the focus here is on the image resolution."),
               br(),
               h4("Image Rotation Tab: "),
               p("This tab like the Image Resolution Tab helps you understand how image rotation affects the model’s performance. It shows an original image along with two versions of the image at 90 degrees and 180 degrees rotation. The validation losses for the different rotations are compared and visualised through interactive histograms."),
               br(),
               h4("Visualisation Tab: "),
               p("This tab allows you to be able to compare the validation loss learning curve of the different models, with or without augmentation. "),
               br(),
               h4("Demonstration Tab:"),
               p("Upload a cell image and see how different augmentation changes your results!"),
               br(),
               
               p("Remember, the goal of this application is to show how different factors can impact a CNN model's performance. While exploring, please consider how these factors might influence the design and implementation of your own models."),
               p("Enjoy exploring!")
             )
    ),
    
    
    tabPanel("Gaussian Noise Levels",
             titlePanel("Effect of Different Gaussian Noise Levels"),
             
             fluidRow(align = "center",
                      
                      column(width = 4, uiOutput(outputId = "original1")),
                      column(width = 4, uiOutput(outputId = "noise_low")),
                      column(width = 4, uiOutput(outputId = "noise_high")),
                  
             ),
             
             fluidRow(style = "padding: 25px; font-size: 17px; margin: auto; max-width: 1000px; max-height: 500px; border: 4px outset; background: ghostwhite",
                      htmlOutput(outputId = "noise_description")),
             
             titlePanel("Validation Loss Comparison"),
             
             fluidRow(
               style = "border: 4px groove;",
               align = "center",
               column(width = 6, plotlyOutput(outputId = "noise_plot")),
               column(width = 6, plotlyOutput(outputId = "noise_catent"))
             )
    ),
    
    tabPanel("Image Resolution",
             titlePanel("Effect of Different Image Resolution"),
             
             fluidRow(align = "center",
                      
                      column(width = 4, uiOutput(outputId = "original2")),
                      column(width = 4, uiOutput(outputId = "resolution_low")),
                      column(width = 4, uiOutput(outputId = "resolution_medium")),
                      ),
             
             fluidRow(style = "padding: 25px; font-size: 17px; margin: auto; max-width: 1000px; max-height: 500px; border: 4px outset; background: ghostwhite",
                      htmlOutput(outputId = "resolution_description")),
             
             titlePanel("Validation Loss Comparison"),
             
             fluidRow(
               style = "border: 4px groove;",
               align = "center",
               column(width = 6, plotlyOutput(outputId = "res_plot")),
               column(width = 6, plotlyOutput(outputId = "res_catent"))
             )
             
    ),
    
    tabPanel("Image Rotation",
             titlePanel("Effect of Different Image Rotation"),
             
             fluidRow(align = "center",
                      
                      column(width = 4, uiOutput(outputId = "original3")),
                      column(width = 4, uiOutput(outputId = "rotate_90")),
                      column(width = 4, uiOutput(outputId = "rotate_180")),
                      
             ),
             
             fluidRow(style = "padding: 25px; font-size: 17px; margin: auto; max-width: 1000px; max-height: 500px; border: 4px outset; background: ghostwhite",
                      htmlOutput(outputId = "rotate_description")),
             
             titlePanel("Validation Loss Comparison"),
             
             fluidRow(
               style = "border: 4px groove;",
               align = "center",
               column(width = 6, plotlyOutput(outputId = "rotate_plot")),
               column(width = 6, plotlyOutput(outputId = "rotate_catent"))
             )
    ),
    
    tabPanel("Visualisation",
             titlePanel("Interactive Learning Curve Visualisation "),
             
             fluidRow(
               column(3, style = "border: 4px groove #922C40;padding:25px; background-color: #ECD5BB;",
                      selectizeInput("model_choice", "Model",
                                     choices = list(`Binary` = "binary",
                                                    `Binary w/class weights` = "binary_cweights",
                                                    `Category` = "catent",
                                                    `Category w/class weights` = "catent_cweights",
                                                    `RMSprop` = "rmsprop",
                                                    `RMSprop w/class weights` = "rmsprop_cweights"), 
                                     selected = list(`Binary` = "binary",
                                                     `Binary w/class weights` = "binary_cweights",
                                                     `Category` = "catent",
                                                     `Category w/class weights` = "catent_cweights",
                                                     `RMSprop` = "rmsprop",
                                                     `RMSprop w/class weights` = "rmsprop_cweights"),
                                     multiple = T),
                      
                      checkboxGroupInput("gaussian_level", "Gaussian Noise",
                                         choices = list(`Low (0.2)` = "noise02",
                                                        `High (0.8)` = "noise08",
                                                        `Random` = "noiserand")),
                      
                      checkboxGroupInput("resolution_choice", "Resolution",
                                         choices = list(`16x16` = "res16",
                                                        `32x32` = "res32",
                                                        `Random` = "res_rand")),
                      
                      checkboxGroupInput("rotation_degrees", "Rotation",
                                         choices = list(`90°` = "rotate90",
                                                        `180°` = "rotate180",
                                                        `Random` = "rotaterand")),
                      
                      checkboxGroupInput("combine_model", "Combined Model",
                                         choices = list(`Combined` = "comb")), 
                      actionButton("reset_vis", "Reset Input")
                      ),
               
               mainPanel(
                 # Interactive plot for binary and rmsprop models
                 column(9, align="center",
                        plotlyOutput("binary_rmsprop"),
                        # Plot for categorical models
                        plotlyOutput("categorical"))
               )
             )
    ),
    
    tabPanel("Demonstration",
             titlePanel("Demo"),
             
             fluidRow(
               column(3, style = "border: 4px groove #922C40;padding:25px; background-color: #ECD5BB;",
                      fileInput("file", h3("File input")),
                      sliderInput("gnoise", label = "Gaussian Noise",
                                  min = 0, max = 1, value = 0.2),
                      sliderInput("demo_rotation", label = "Rotation",
                                  min = 0, max = 359, value = 0),
                      radioButtons("demo_resolution", label = "Resolution",
                                   choices = list ("Default (64x64)" = 64, "16x16" = 16, "32x32" = 32),
                                   selected = 64),
                      actionButton("reset_input", "Reset Input")
               ),
               mainPanel(
                 # Output: Histogram ----
                 plotOutput(outputId = "image"),
                 titlePanel("Output"),
                 htmlOutput(style = "padding: 25px; font-size: 17px; margin: auto; border: 4px outset; background: #CBE1DF",
                            outputId = "prediction"))
               
             )
    )
  )
)


server <- function(input, output, session) {
  
  cell_image <- reactive({
    req(input$file)
    EBImage::readImage(input$file$datapath)
  })
  
  
  output$original1 <- renderUI({
    
    tags$figure (
      tags$img(
        src = "original.png",
        width = 300,
        height = 300,
        alt = "Original Image with no noise"
      ),
      tags$figcaption("Original Image")
    )
    
  })
  
  output$original2 <- renderUI({
    
    tags$figure (
      tags$img(
        src = "original.png",
        width = 300,
        height = 300,
        alt = "Original Image with no noise"
      ),
      tags$figcaption("Original Image")
    )
    
  })
  
  output$noise_low <- renderUI({
    
    tags$figure (
      tags$img(
        src = "noise02_example.png",
        width = 300,
        height = 300,
        alt = "Image with low noise"
      ),
      tags$figcaption("Low Noise (0.2)")
    )
    
  })
  
  output$noise_high <- renderUI({
    
    tags$figure (
      tags$img(
        src = "noise08_example.png",
        width = 300,
        height = 300,
        alt = "Image with high noise"
      ),
      tags$figcaption("High Noise (0.8)")
    )
    
  })
  
  output$noise_description <- renderText({
    
    HTML("Adding noise expands the size of the training dataset. Each time a training sample is exposed to the model, 
    random noise is added to the input variables making them different every time it is exposed to the model 
    (<a href='https://www.academia.edu/38223830/Adaptive_Computation_and_Machine_Learning_series_Deep_learning_The_MIT_Press_2016_pdf'>Maulana, n.d.</a>). 
    In addition, from scientific research, we found that the addition of noise to the input 
    data of a neural network during training can lead to significant improvements in generalisation performance 
    (<a href='https://doi.org/10.1162/neco.1995.7.1.108'>Bishop, 1995</a>).")
    
  })
  
  output$noise_plot <- renderPlotly({
    level_order <- c("none", "low", "high", "random")
    
    gaussian <- val_loss |> filter((noise_type == "gaussian" | noise_type == "none") & !grepl("Category", model))
    
    ggplotly(ggplot(data = gaussian, aes(x = factor(noise_level, level = level_order), y = val_loss, fill = model)) +
               geom_bar(stat = "identity", position="dodge") +
               xlab("Gaussian Noise Level") + ylab("Validation Loss") +
               ggtitle("Comparison of Validation Loss on Binary and RMSprop Model"))
    
  })
  
  output$noise_catent <- renderPlotly({
    
    level_order <- c("none", "low", "high", "random")
    
    gaussian_catent <- val_loss |> filter((noise_type == "gaussian" | noise_type == "none") & grepl("Category", model))
    
    ggplotly(ggplot(data = gaussian_catent, aes(x = factor(noise_level, level = level_order), y = val_loss, fill = model)) + 
               geom_bar(stat = "identity", position="dodge") +
               xlab("Gaussian Noise Level") + ylab("Validation Loss") +
               ggtitle("Comparison of Validation Loss on Categorical Model"))
    
  })
  
  output$resolution_description <- renderText({
    "Examining image resolution is crucial in deep learning applications, particularly in tasks involving radiology 
     (like chest radiographic image analysis).  While large image resolutions provide more information for 
     classification, studies have shown that better model performance can sometimes be achieved with lower image 
     resolutions (<a href='https://doi.org/10.1148/ryai.2019190015'>Sabottke & Spieler, 2020</a>). 
     This is because reducing the number of inputs or features can help 
     minimise the number of parameters to be optimsed, thereby decreasing the risk of the model over-fitting. 
     However, excessively reducing image resolution can lead to the elimination of important information useful 
     for classification. Hence, we decided to look into how low, high, and random resolutions have an effect on 
     the classification model."

  })
  
  output$resolution_low <- renderUI({
    
    tags$figure (
      tags$img(
        src = "res16_example.png",
        width = 300,
        height = 300,
        alt = "Image with 16x16 resolution"
      ),
      tags$figcaption("16x16 Resolution")
    )
    
  })
  
  output$resolution_medium <- renderUI({
    
    tags$figure (
      tags$img(
        src = "res32_example.png",
        width = 300,
        height = 300,
        alt = "Image with 32x32 resolution"
      ),
      tags$figcaption("32x32 Resolution")
    )
    
  })
  
  output$res_plot <- renderPlotly({
    level_order <- c("none", "low", "medium", "random")
    
    resolution <- val_loss |> filter((noise_type == "resolution" | noise_type == "none") & !grepl("Category", model))
    
    ggplotly(ggplot(data = resolution, aes(x = factor(noise_level, level = level_order), y = val_loss, fill = model)) +
               geom_bar(stat = "identity", position="dodge") +
               xlab("Resolution") + ylab("Validation Loss") +
               ggtitle("Comparison of Validation Loss on Binary and RMSprop Model"))
    
  }) 
  
  output$res_catent <- renderPlotly({
    
    level_order <- c("none", "low", "medium", "random")
    
    resolution_catent <- val_loss |> filter((noise_type == "resolution" | noise_type == "none") & grepl("Category", model))
    
    ggplotly(ggplot(data = resolution_catent, aes(x = factor(noise_level, level = level_order), y = val_loss, fill = model)) + 
               geom_bar(stat = "identity", position="dodge") +
               xlab("Resolution") + ylab("Validation Loss") +
               ggtitle("Comparison of Validation Loss on Categorical Model"))
    
  })
  
  output$original3 <- renderUI({
    
    tags$figure (
      tags$img(
        src = "original.png",
        width = 300,
        height = 300,
        alt = "Original Image with no rotation"
      ),
      tags$figcaption("Original Image")
    )
    
  })
  
  output$rotate_90 <- renderUI({
    
    tags$figure (
      tags$img(
        src = "rotate90_example.png",
        width = 300,
        height = 300,
        alt = "Image with 90 degree rotation"
      ),
      tags$figcaption("90 Degree Rotation")
    )
    
  })
  
  output$rotate_180 <- renderUI({
    
    tags$figure (
      tags$img(
        src = "rotate180_example.png",
        width = 300,
        height = 300,
        alt = "Image with 180 degree rotation"
      ),
      tags$figcaption("180 Degree Rotation")
    )
    
  })
  
  output$rotate_description <- renderText({
    
    "Image rotation is a critical data augmentation strategy in improving the performance of machine learning models.
    Traditional methods like image rotations, are fast and straightforward to implement and have proven to be 
    effective in increasing the training dataset (<a href = 'https://doi.org/10.1109/IIPHDW.2018.8388338'>Mikołajczyk & Grochowski, 2018</a>). 
    By rotating an image at 
    various angles, the model is exposed to more diverse observations of the cells, enhancing its ability to 
    generalise across different perspectives (<a href='https://www.amygb.ai/blog/what-is-data-augmentation-in-image-processing'>What Is Data Augmentation in Image Processing?, n.d.</a>). 
    Data augmentation can quickly expand the existing dataset by adding more relevant observations, which is highly 
    beneficial as deep learning models require larger datasets for training."
    
  })
  
  output$rotate_plot <- renderPlotly({
    level_order <- c("none", "low", "high", "random")
    
    resolution <- val_loss |> filter((noise_type == "rotation" | noise_type == "none") & !grepl("Category", model))
    
    ggplotly(ggplot(data = resolution, aes(x = factor(noise_level, level = level_order), y = val_loss, fill = model)) +
               geom_bar(stat = "identity", position="dodge") +
               xlab("Rotation") + ylab("Validation Loss") +
               ggtitle("Comparison of Validation Loss on Binary and RMSprop Model"))
    
  }) 
  
  output$rotate_catent <- renderPlotly({
    
    level_order <- c("none", "low", "high", "random")
    
    resolution_catent <- val_loss |> filter((noise_type == "rotation" | noise_type == "none") & grepl("Category", model))
    
    ggplotly(ggplot(data = resolution_catent, aes(x = factor(noise_level, level = level_order), y = val_loss, fill = model)) + 
               geom_bar(stat = "identity", position="dodge") +
               xlab("Rotation") + ylab("Validation Loss") +
               ggtitle("Comparison of Validation Loss on Categorical Model"))
    
  })
  
  output$binary_rmsprop <- renderPlotly({
    
    model_ls = c("binary", "binary_cweights", "rmsprop", "rmsprop_cweights")
    model_choice = input$model_choice[!(input$model_choice %in% c("catent", "catent_cweights"))]
    # Data Frame to keep add data of all the models
    model_df <- NULL
    
    if (!is.null(model_choice)){
      
      for (model in model_choice) {
        
        df <- read.csv(paste0("models/cnn_", model, "/training.csv"))
        if (model == "binary") {
          df$model = "Binary"
        } else if (model == "binary_cweights") {
          df$model = "Binary Cweights"
        } else if (model == "rmsprop") {
          df$model = "RMSprop"
        } else if(model == "rmsprop_cweights") {
          df$model = "RMSprop Cweights"
        }
        df$noise <- "No Noise"
        model_df <- rbind(model_df, df)
      }
    }
    
    if (!is.null(input$gaussian_level)) {
      
      if (!is.null(model_choice)){
        
        for (model in model_choice) {
          for (noise in input$gaussian_level){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "binary") {
              df$model = "Binary"
            } else if (model == "binary_cweights") {
              df$model = "Binary Cweights"
            } else if (model == "rmsprop") {
              df$model = "RMSprop"
            } else if(model == "rmsprop_cweights") {
              df$model = "RMSprop Cweights"
            }
            
            if (noise == "noise02") {
              df$noise <- "Low Gaussian"
            } else if (noise == "noise08") {
              df$noise <- "High Gaussian" 
            } else if (noise == "noiserand") {
              df$noise <- "Random Gaussian"
            }
            model_df <- rbind(model_df, df)
          }
        }
        
      } else {
        # Compare all models of noise level
        for (model in model_ls) {
          for (noise in input$gaussian_level){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "binary") {
              df$model = "Binary"
            } else if (model == "binary_cweights") {
              df$model = "Binary Cweights"
            } else if (model == "rmsprop") {
              df$model = "RMSprop"
            } else if(model == "rmsprop_cweights") {
              df$model = "RMSprop Cweights"
            }
            
            if (noise == "noise02") {
              df$noise <- "Low Gaussian"
            } else if (noise == "noise08") {
              df$noise <- "High Gaussian" 
            } else if (noise == "noiserand") {
              df$noise <- "Random Gaussian"
            }
            model_df <- rbind(model_df, df)
          }
          
        }
      }
    }
    
    if (!is.null(input$rotation_degrees)) {
      
      if (!is.null(model_choice)){
        
        for (model in model_choice) {
          for (noise in input$rotation_degrees){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "binary") {
              df$model = "Binary"
            } else if (model == "binary_cweights") {
              df$model = "Binary Cweights"
            } else if (model == "rmsprop") {
              df$model = "RMSprop"
            } else if(model == "rmsprop_cweights") {
              df$model = "RMSprop Cweights"
            }
            
            if (noise == "rotate90") {
              df$noise <- "90"
            } else if (noise == "rotate180") {
              df$noise <- "180" 
            } else if (noise == "rotaterand") {
              df$noise <- "Random Rotation"
            }
            model_df <- rbind(model_df, df)
          }
        }
        
      } else {
        # Compare all models of noise level
        for (model in model_ls) {
          for (noise in input$rotation_degrees){
           
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "binary") {
              df$model = "Binary"
            } else if (model == "binary_cweights") {
              df$model = "Binary Cweights"
            } else if (model == "rmsprop") {
              df$model = "RMSprop"
            } else if(model == "rmsprop_cweights") {
              df$model = "RMSprop Cweights"
            }
            
            if (noise == "rotate90") {
              df$noise <- "90"
            } else if (noise == "rotate180") {
              df$noise <- "180" 
            } else if (noise == "rotaterand") {
              df$noise <- "Random Rotation"
            }
            model_df <- rbind(model_df, df)
          }
          
        }
      }
    }
    
    if (!is.null(input$resolution_choice)) {
      
      if (!is.null(model_choice)){
        
        for (model in model_choice) {
          for (noise in input$resolution_choice){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "binary") {
              df$model = "Binary"
            } else if (model == "binary_cweights") {
              df$model = "Binary Cweights"
            } else if (model == "rmsprop") {
              df$model = "RMSprop"
            } else if(model == "rmsprop_cweights") {
              df$model = "RMSprop Cweights"
            }
            
            if (noise == "res16") {
              df$noise <- "Low Resolution"
            } else if (noise == "res32") {
              df$noise <- "Medium Resolution" 
            } else if (noise == "res_rand") {
              df$noise <- "Random Resolution"
            }
            model_df <- rbind(model_df, df)
          }
        }
        
      } else {
        # Compare all models of noise level
        for (model in model_ls) {
          for (noise in input$resolution_choice){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "binary") {
              df$model = "Binary"
            } else if (model == "binary_cweights") {
              df$model = "Binary Cweights"
            } else if (model == "rmsprop") {
              df$model = "RMSprop"
            } else if(model == "rmsprop_cweights") {
              df$model = "RMSprop Cweights"
            }
            
            if (noise == "res16") {
              df$noise <- "Low Resolution"
            } else if (noise == "res32") {
              df$noise <- "Medium Resolution" 
            } else if (noise == "res_rand") {
              df$noise <- "Random Resolution"
            }
            model_df <- rbind(model_df, df)
          }
          
        }
      }
    }
    
    if (!is.null(input$combine_model)) {
      
      if (!is.null(model_choice)){
        
        for (model in model_choice) {
          for (noise in input$combine_model){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "binary") {
              df$model = "Binary"
            } else if (model == "binary_cweights") {
              df$model = "Binary Cweights"
            } else if (model == "rmsprop") {
              df$model = "RMSprop"
            } else if(model == "rmsprop_cweights") {
              df$model = "RMSprop Cweights"
            }
            
            df$noise <- "Combined"
            model_df <- rbind(model_df, df)
          }
        }
        
      } else {
        # Compare all models of noise level
        for (model in model_ls) {
          for (noise in input$combine_model){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "binary") {
              df$model = "Binary"
            } else if (model == "binary_cweights") {
              df$model = "Binary Cweights"
            } else if (model == "rmsprop") {
              df$model = "RMSprop"
            } else if(model == "rmsprop_cweights") {
              df$model = "RMSprop Cweights"
            }
            
            df$noise <- "Combined"
            model_df <- rbind(model_df, df)
          }
          
        }
      }
    }
    
    
    if (is.null(model_df)){
      plot(1, xlab = "",
           ylab = "", xlim = c(0, 100), 
           ylim = c(0, 0.5), axes = F)
      
      box(bty="l")
      axis(2)
      axis(1)
    } else {
      
      ggplotly(ggplot(data = model_df, aes(x = X, y = val_loss, colour = interaction(model, noise))) + 
        geom_line() + xlab("Epoch") + ylab("Validation Loss") + 
        labs(title = "Model with Binary Cross-Entropy and RMSprop Optimiser", color='Model'), width = 1000)
    }

    
  })
  
  output$categorical <- renderPlotly({
    
    model_ls = c("catent", "catent_cweights")
    model_choice = input$model_choice[(input$model_choice %in% c("catent", "catent_cweights"))]
    # Data Frame to keep add data of all the models
    model_df <- NULL
    
    if (!is.null(model_choice)){
      
      for (model in model_choice) {
        
        df <- read.csv(paste0("models/cnn_", model, "/training.csv"))
        if (model == "catent") {
          df$model = "Categorical"
        } else if (model == "catent_cweights") {
          df$model = "Categorical Cweights"
        } 
        
        df$noise <- "No Noise"
        model_df <- rbind(model_df, df)
      }
    }
    
    if (!is.null(input$gaussian_level)) {
      
      if (!is.null(model_choice)){
        
        for (noise in input$gaussian_level) {
          for (model in model_choice){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "catent") {
              df$model = "Categorical"
            } else if (model == "catent_cweights") {
              df$model = "Categorical Cweights"
            } 
            
            if (noise == "noise02") {
              df$noise <- "Low Gaussian"
            } else if (noise == "noise08") {
              df$noise <- "High Gaussian" 
            } else if (noise == "noiserand") {
              df$noise <- "Random Gaussian"
            }
            
            model_df <- rbind(model_df, df)
          }
        }
        
      } else {
        # Compare all models of noise level
        for (noise in input$gaussian_level) {
          for (model in model_ls){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "catent") {
              df$model = "Categorical"
            } else if (model == "catent_cweights") {
              df$model = "Categorical Cweights"
            } 
            
            if (noise == "noise02") {
              df$noise <- "Low Gaussian"
            } else if (noise == "noise08") {
              df$noise <- "High Gaussian" 
            } else if (noise == "noiserand") {
              df$noise <- "Random Gaussian"
            }
            
            model_df <- rbind(model_df, df)
          }
          
        }
      }
    }
    
    if (!is.null(input$rotation_degrees)) {
      
      if (!is.null(model_choice)){
        
        for (noise in input$rotation_degrees) {
          for (model in model_choice){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "catent") {
              df$model = "Categorical"
            } else if (model == "catent_cweights") {
              df$model = "Categorical Cweights"
            } 
            
            if (noise == "rotate90") {
              df$noise <- "90"
            } else if (noise == "rotate180") {
              df$noise <- "180" 
            } else if (noise == "rotaterand") {
              df$noise <- "Random Rotation"
            }
            model_df <- rbind(model_df, df)
          }
        }
        
      } else {
        # Compare all models of noise level
        for (noise in input$rotation_degrees) {
          for (model in model_ls){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "catent") {
              df$model = "Categorical"
            } else if (model == "catent_cweights") {
              df$model = "Categorical Cweights"
            }
            
            if (noise == "rotate90") {
              df$noise <- "90"
            } else if (noise == "rotate180") {
              df$noise <- "180" 
            } else if (noise == "rotaterand") {
              df$noise <- "Random Rotation"
            }
            model_df <- rbind(model_df, df)
          }
          
        }
      }
    }
    
    if (!is.null(input$resolution_choice)) {
      
      if (!is.null(model_choice)){
        
        for (noise in input$resolution_choice) {
          for (model in model_choice){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "catent") {
              df$model = "Categorical"
            } else if (model == "catent_cweights") {
              df$model = "Categorical Cweights"
            }
            
            if (noise == "res16") {
              df$noise <- "Low Resolution"
            } else if (noise == "res32") {
              df$noise <- "Medium Resolution" 
            } else if (noise == "res_rand") {
              df$noise <- "Random Resolution"
            }
            model_df <- rbind(model_df, df)
          }
        }
        
      } else {
        # Compare all models of noise level
        for (noise in input$resolution_choice) {
          for (model in model_ls){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "catent") {
              df$model = "Categorical"
            } else if (model == "catent_cweights") {
              df$model = "Categorical Cweights"
            }
            
            if (noise == "res16") {
              df$noise <- "Low Resolution"
            } else if (noise == "res32") {
              df$noise <- "Medium Resolution" 
            } else if (noise == "res_rand") {
              df$noise <- "Random Resolution"
            }
            model_df <- rbind(model_df, df)
          }
          
        }
      }
    }
    
    if (!is.null(input$combine_model)) {
      
      if (!is.null(model_choice)){
        
        for (noise in input$combine_model) {
          for (model in model_choice){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "catent") {
              df$model = "Categorical"
            } else if (model == "catent_cweights") {
              df$model = "Categorical Cweights"
            }
            
            df$noise <- "Combined"
            model_df <- rbind(model_df, df)
          }
        }
        
      } else {
        # Compare all models of noise level
        for (noise in input$combine_model) {
          for (model in model_ls){
            
            df <- read.csv(paste0("models/cnn_", model, "_", noise, "/training.csv"))
            if (model == "catent") {
              df$model = "Categorical"
            } else if (model == "catent_cweights") {
              df$model = "Categorical Cweights"
            }
            
            df$noise <- "Combined"
            model_df <- rbind(model_df, df)
          }
          
        }
      }
    }
    
    if (is.null(model_df)){
      plot(1, xlab = "",
           ylab = "", xlim = c(0, 100), 
           ylim = c(0, 0.5), axes = F)
      
      box(bty="l")
      axis(2)
      axis(1)
    } else {
      
      ggplotly(ggplot(data = model_df, aes(x = X, y = val_loss, colour = interaction(model, noise))) + 
        geom_line() + xlab("Epoch") + ylab("Validation Loss") + 
        labs(title = "Validation Loss of Model with Categorical Cross-Entropy", colour = "Model"), width = 1000)
    }
    
    
  })
  
  observeEvent(input$reset_input, {
    updateSliderInput(session,"gnoise", value = 0.2)          
    updateSliderInput(session,"demo_rotation", value = 0)
    updateRadioButtons(session,"demo_resolution", selected = 64 )
    
  })
  
  add_gaussian_noise_display <- function(image, mean = 0, sd = 0.1) {
    set.seed(6)
    img <- image + rnorm(1, mean, sd)
    return(img)
  }
  
  output$prediction <- renderText({
    
    x <- array(dim = c(1, 64, 64, 1))
    x[1,,,1] <- resize(rotate(cell_image(), as.integer(input$demo_rotation)), as.integer(input$demo_resolution), as.integer(input$demo_resolution))
    x <- x - mean(x)
    x <- add_gaussian_noise_display(x, mean=as.integer(input$gnoise))
    x <- array_reshape(x, dim = c(1, 64, 64, 1))
    pred <- predict(model, x)
    pred_class = which.max(pred)
  
  paste0("The predicted cluster is <b>", pred_class, "</b>.")
  })
  
  output$image <- renderPlot({
  
    #img <- resize(image(), 64, 64)
    img <- add_gaussian_noise_display(cell_image(), mean = input$gnoise)
    img <- rotate(img, input$demo_rotation)
    img <- resize(img, as.integer(input$demo_resolution), as.integer(input$demo_resolution))
    display(img, method = "raster")
  })
  
  observeEvent(input$reset_vis, {
    updateSelectizeInput(session,"model_choice",
                         choices = list(`Binary` = "binary",
                                        `Binary w/class weights` = "binary_cweights",
                                        `Category` = "catent",
                                        `Category w/class weights` = "catent_cweights",
                                        `RMSprop` = "rmsprop",
                                        `RMSprop w/class weights` = "rmsprop_cweights"), 
                         selected = list(`Binary` = "binary",
                                         `Binary w/class weights` = "binary_cweights",
                                         `Category` = "catent",
                                         `Category w/class weights` = "catent_cweights",
                                         `RMSprop` = "rmsprop",
                                         `RMSprop w/class weights` = "rmsprop_cweights")) 
    
    updateCheckboxGroupInput(session,"gaussian_level", choices = list(`Low (0.2)` = "noise02",
                                                                      `High (0.8)` = "noise08",
                                                                      `Random` = "noiserand"))
    updateCheckboxGroupInput(session,"rotation_degrees", choices = list(`90°` = "rotate90",
                                                                        `180°` = "rotate180",
                                                                        `Random` = "rotaterand"))
    updateCheckboxGroupInput(session,"resolution_choice", choices = list(`16x16` = "res16",
                                                                         `32x32` = "res32",
                                                                         `Random` = "res_rand"))
    
  })
  
  
  
}

shinyApp(ui = ui, server = server)


