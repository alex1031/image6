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
model <- load_model_tf("models/cnn")
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
               column(4,
                      selectizeInput("model_choice", "Model",
                                     choices = list(`Binary` = "bin",
                                                    `Binary w/class weights` = "bincweights",
                                                    `Category` = "cat",
                                                    `Category w/class weights` = "catcweights",
                                                    `RMSprop` = "rms",
                                                    `RMSprop w/class weights` = "rmscweights"), multiple = T))
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
                      actionButton("reset_input", "Reset inputs")
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
               ggtitle("Comparison of Validation Loss on Model with Binary Cross-Entropy Loss and RMSprop Optimiser"))
    
  })
  
  output$noise_catent <- renderPlotly({
    
    level_order <- c("none", "low", "high", "random")
    
    gaussian_catent <- val_loss |> filter((noise_type == "gaussian" | noise_type == "none") & grepl("Category", model))
    
    ggplotly(ggplot(data = gaussian_catent, aes(x = factor(noise_level, level = level_order), y = val_loss, fill = model)) + 
               geom_bar(stat = "identity", position="dodge") +
               xlab("Gaussian Noise Level") + ylab("Validation Loss") +
               ggtitle("Comparison of Validation Loss on Model with Categorical Cross-Entropy Loss"))
    
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
  
  observeEvent(input$reset_input, {
    updateSliderInput(session,"gnoise", value = 0.2)          
    updateSliderInput(session,"demo_rotation", value = 0)
    updateRadioButtons(session,"demo_resolution", selected = 64 )
    
  })
  
  
  
}

shinyApp(ui = ui, server = server)


