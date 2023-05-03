#
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
               h1("introduction")
             )),
    
    tabPanel("Noise Levels",
             sidebarLayout(
               sidebarPanel(
                 sliderInput("noise-level", "Noise Levels:", min = 0, max = 1, value = 0.2)
               ),
               mainPanel(
                 tabsetPanel(
                   tabPanel("Performance"),
                   tabPanel("Mutated Image")
                 )
               )
             )
    ), 
    
    tabPanel("Image Resolution",
             sidebarLayout(
               sidebarPanel(
                 sliderInput("resolution", "Resolution:", min = 1, max = 150, value = 55)
               ),
               mainPanel(
                 tabsetPanel(
                   tabPanel("Performance"),
                   tabPanel("Mutated Image")
                 )
               )
             )
    ), 
    
    tabPanel("Image Rotation",
             sidebarLayout(
               sidebarPanel(
                 sliderInput("rotation", "Rotation:", min = 0, max = 360, value = 60)
               ),
               mainPanel(
                 tabsetPanel(
                   tabPanel("Performance"),
                   tabPanel("Mutated Image")
                 )
               )
             )
    ),
    
    tabPanel("Demonstration",
             fluidRow(
               column(4,
                      fileInput("file", h3("File input")), 
                      sliderInput("noise-level-demo", "Noise Level", min = 0, max = 1, value = 0.2),
                      sliderInput("resolution-demo", "Resolution", min = 1, max = 150, value = 55),
                      sliderInput("rotation-demo", "Rotation", min = 0, max = 360, value = 60)
                      ),
               actionButton("change", "change")
              
             )
    )
  )
)

server <- function(input, output) {
  randomVals <- eventReactive(input$change, {
    runif(input$file)
  })
  

  
}

shinyApp(ui = ui, server = server)


