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
library(keras)
library(EBImage)

# Load the model
model <- load_model_tf("../models/cnn_catent_cweights_comb")


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
    
    tabPanel("Noise Levels",
             sidebarLayout(
               sidebarPanel(
                 sliderInput("noise-level", "Noise Levels:", min = 0, max = 1, value = 0.2)
               ),
               mainPanel(
                 # 6 learning model 
                 
               )
             )
    ),
    
    tabPanel("Image Resolution",
             sidebarLayout(
               sidebarPanel(
                 sliderInput("resolution", "Resolution:", min = 1, max = 150, value = 55)
               ),
               mainPanel(
                 # Output: Histogram ----
                 # 6 learning model 
                 #textOutput(outputId = "prediction"),
                 #plotOutput(outputId = "image")
               )
             )
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
    
    tabPanel("Demonstration",
             fluidRow(
               column(4,
                      fileInput("file", h3("File input")),
                      sliderInput("gnoise", label = "Gaussian Noise",
                                  min = 0, max = 1, value = 0.2)
               ),
               mainPanel(
                 # Output: Histogram ----
                 plotOutput(outputId = "image"),
                 textOutput(outputId = "prediction"),
                 actionButton("change", "change"))
               
             )
    )
  )
)


server <- function(input, output) {
  
  add_gaussian_noise <- function(images, mean = 0, sd = 0.1) {
    noisy_images <- array(0, dim = dim(images))
    for (i in 1:nrow(images)) {
      noisy_images[i,] <- images[i,,,] + rnorm(1, mean, sd,)
    }
    return(noisy_images)
  }
  
  randomVals <- eventReactive(input$change, {
    runif(input$file)
    
  })
  
  image <- reactive({
    req(input$file)
    png::readPNG(input$file$datapath)
  })
  
  output$prediction <- renderText({
    
    img <- resize(image(), 64, 64)
    img_copy <- img
    img <- array_reshape(img, dim = c(1, 64, 64, 1))
    img <- img - mean(img)
    
    pred <- model |> predict(img)
    pred_class = which.max(pred)
    ## 
    paste0("The predicted class number is ", pred_class)
  })
  
  output$image <- renderPlot({
    display <- add_gaussian_noise(img_copy, mean = input$gnoise)
    plot(as.raster(display))
  })
  
  
  
}

shinyApp(ui = ui, server = server)


