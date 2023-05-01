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
#library(keras)
library(EBImage)

# Load the model
#model <- load_model_tf("cnn/")


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
               ),
               mainPanel(
                 # Output: Histogram ----
                 textOutput(outputId = "prediction"),
                 plotOutput(outputId = "image"),
                 
               actionButton("change", "change"))
               
             )
    )
  )
)



server <- function(input, output) {
  
  randomVals <- eventReactive(input$change, {
    runif(input$file)
    
  })
    
    image <- reactive({
      req(input$file)
      png::readPNG(input$file$datapath)
      
    })
    
    output$prediction <- renderText({
      
      img <- image() %>% 
        array_reshape(., dim = c(1, dim(.), 1))
      
      paste0("The predicted class number is ", predict_classes(model, img))
    })
    
    output$image <- renderPlot({
      plot(as.raster(image()))
    })
    
  
  
}

shinyApp(ui = ui, server = server)


