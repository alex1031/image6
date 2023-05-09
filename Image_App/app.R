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
library(tensorflow)
model <- load_model_tf("models/cnn/")


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
                                  min = 0, max = 1, value = 0.2),
                      sliderInput("demo_rotation", label = "Rotation",
                                  min = 0, max = 359, value = 0),
                      radioButtons("demo_resolution", label = "Resolution",
                                  choices = list ("Default (64x64)" = 64, "16x16" = 16, "32x32" = 32),
                                  selected = 64)
               ),
               mainPanel(
                 # Output: Histogram ----
                 plotOutput(outputId = "image"),
                 textOutput(outputId = "prediction"))
               
             )
    )
  )
)


server <- function(input, output) {
  
  add_gaussian_noise <- function(image, mean = 0, sd = 0.1) {
    set.seed(6)
    img <- image[1,,,] + rnorm(1, mean, sd)
    return(img)
  }
  
  image <- reactive({
    req(input$file)
    png::readPNG(input$file$datapath)
  })
  
  #output$prediction <- renderText({
    
    #img <- resize(image(), 64, 64)
    #img <- array_reshape(img, dim = c(1, 64, 64, 1))
    #img <- img - mean(img)
    
    #pred <- predict(model, img)
    #pred_class = which.max(pred)
    ## 
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
  
  
  
}

shinyApp(ui = ui, server = server)


