#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(rdrop2)

# Define UI for application 
ui <- fluidPage(
  
  titlePanel("Robustness on Cell Image Classification"),
  
  navlistPanel(
    "Affects of Robustness",
    tabPanel("Noise Levels"),
    tabPanelBody(
      sidebarLayout(
        sidebarPanel(
          sliderInput("bins",
                      "Noise Levels:",
                      min = 0,
                      max = 1,
                      value = 0.2)
        )
      
    )
    ), 
    mainPanel(
      tabsetPanel(
        tabPanel("Performance", plotOutput("plot")), 
        tabPanel("Mutated Image", plotOutput("summary")), 
      )
    ),
    
    
    tabPanel("Image Resolution"),
    tabPanelBody(
      sidebarLayout(
        sidebarPanel(
          sliderInput("bins",
                      "Resolution:",
                      min = 1,
                      max = 150,
                      value = 55)
        )
        
      )
    ), 
    mainPanel(
      tabsetPanel(
        tabPanel("Performance", plotOutput("plot")), 
        tabPanel("Mutated Image", plotOutput("summary")), 
      )
    ),
    
    
    tabPanel("Image Rotation"),
    tabPanelBody(
      sidebarLayout(
        sidebarPanel(
          sliderInput("bins",
                      "Rotation:",
                      min = 0,
                      max = 360,
                      value = 60)
        )
        
      )
    ), 
    mainPanel(
      tabsetPanel(
        tabPanel("Performance", plotOutput("plot")), 
        tabPanel("Mutated Image", plotOutput("summary")), 
      )
    ),
    
    
    "Demonstration",
    tabPanel("...")
  )
)





# Define server logic required to draw a histogram 
# Serves = analysis to build the app
server <- function(input, output) {

    output$distPlot <- renderPlot({
        # generate bins based on input$bins from ui.R
        x    <- faithful[, 2]
        bins <- seq(min(x), max(x), length.out = input$bins + 1)

        # draw the histogram with the specified number of bins
        hist(x, breaks = bins, col = 'darkgray', border = 'white',
             xlab = 'Waiting time to next eruption (in mins)',
             main = 'Histogram of waiting times')
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
