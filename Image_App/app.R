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
#model <- load_model_tf("models/cnn")
val_loss <- read.csv("val_loss.csv")

ui <- fluidPage(
  
  titlePanel("Robustness on Cell Image Classification"),
  
  navbarPage(
    "Effects of Robustness",
    
    tags$head(
      tags$style(HTML("
      /* Set the width and height of the app */
      .shiny-output {
        width: 1000px;
        height: 800px;
      }
    "))
    ),
    
    
    tabPanel("Introduction",
             align = "center",
             mainPanel(
             )),
    
    tabPanel("Gaussian Noise Levels",
             titlePanel("Effect of Different Gaussian Noise Levels"),
             
             fluidRow(align = "center",
                      
                      column(width = 4, uiOutput(outputId = "original1")),
                      column(width = 4, uiOutput(outputId = "noise_low")),
                      column(width = 4, uiOutput(outputId = "noise_high")),
                      textOutput(outputId = "noise_description")
                      
             ),
             
             titlePanel("Validation Loss Comparison"),
             
             fluidRow(
               style = "border: 4px groove;",
               align = "center",
               column(width = 6, plotlyOutput(outputId = "noise_plot")),
               column(width = 6, plotlyOutput(outputId = "noise_catent")),
               textOutput(outputId = "noise_analysis")
             )
    ),
    
    tabPanel("Image Resolution",
             titlePanel("Effect of Different Image Resolution"),
             
             fluidRow(align = "center",
                      
                      column(width = 4, uiOutput(outputId = "original2")),
                      column(width = 4, uiOutput(outputId = "resolution_low")),
                      column(width = 4, uiOutput(outputId = "resolution_medium"))
                      
             ),
             
             textOutput(outputId = "resolution_description")
    ),
    
    tabPanel("Image Rotation",
             sidebarLayout(
               sidebarPanel(
                 sliderInput("rotation", "Rotation:", min = 0, max = 360, value = 60)
               ),
               mainPanel(
                 # Output: Histogram ----
                 #textOutput(outputId = "prediction"),
                 #plotOutput(outputId = "image")
               )
             )
    ),
    
    tabPanel("Visualisation",
             titlePanel("Interactive Visualisation Learning Curve"),
             
             fluidRow(
               column(3,
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
                      checkboxGroupInput("rotation_degrees", "Rotation",
                                         choices = list(`90°` = "rotate90",
                                                        `180°` = "rotate180",
                                                        `Random` = "rotaterand")),
                      checkboxGroupInput("resolution_choice", "Resolution",
                                         choices = list(`16x16` = "res16",
                                                        `32x32` = "res32",
                                                        `Random` = "res_rand")),
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
               column(4,
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
                 textOutput(outputId = "prediction"))
               
             )
    )
  )
)


server <- function(input, output, session) {
  
  add_gaussian_noise <- function(image, mean = 0, sd = 0.1) {
    set.seed(6)
    img <- image[1,,,] + rnorm(1, mean, sd)
    return(img)
  }
  
  image <- reactive({
    req(input$file)
    png::readPNG(input$file$datapath)
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
    
    print("Insert Description.")
    
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
  
  output$noise_analysis <- renderText({
    print("Insert Analysis")
  })
  
  output$resolution_description <- renderText({print("Insert Description.")})
  
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
  
  #output$prediction <- renderText({
  
  #img <- resize(image(), 64, 64)
  #img <- array_reshape(img, dim = c(1, 64, 64, 1))
  #img <- img - mean(img)
  
  #pred <- predict(model, img)
  #pred_class = which.max(pred)
  
  #paste0("The predicted class number is ", summary(model))
  #})
  
  output$image <- renderPlot({
    img <- resize(image(), 64, 64)
    img <- array_reshape(img, dim = c(1, 64, 64, 1))
    img <- add_gaussian_noise(img, mean = input$gnoise)
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


