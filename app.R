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
model <- load_model_tf("/Users/swyi/Desktop/image6-main/cnn")


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
                 plotOutput(outputId = "image"),
                 textOutput(outputId = "prediction"),
                 actionButton("change", "change"))
                 
               )
             )
    )
)


server <- function(input, output) {
  
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
  
  randomVals <- eventReactive(input$change, {
    runif(input$file)
    
  })
  
  image <- reactive({
    req(input$file)
    png::readPNG(input$file$datapath)
  })
  
  output$prediction <- renderText({
  
  img <- image() %>% 
  array_reshape(., dim = c(1, 64, 64, 1))
  
  img <- mask_resize(image(), image(), w=64, h=64)
  img <- array_reshape(img(), dim = c(1, 64, 64, 1))
  
  
  ## 
  paste0("The predicted class number is ", predict(model, img))
  })
  
  output$image <- renderPlot({
    plot(as.raster(image()))
  })
  
  
  
}

shinyApp(ui = ui, server = server)

