## This stores the R codes doing web scrapping on a pet sale website and generating a dataset with listing names, prices, details, and locations (stored as csv).

library(rvest)
library(stringr)
# Scraping listing name, and the auction details
target_url <- "https://www.preloved.co.uk/classifieds/pets/dogs/"

target_page<- read_html(target_url)

# using the pagination information to identify the total number of pages
num_pages <- target_page |> 
  html_nodes('.pagination__control') |> 
  html_attr("href")

# extract the numeric values from the page and select the max value while ignoring any non-numeric characters ("NAs")
total_pages <- str_extract(num_pages, "\\d+")
total_pages <- max(as.numeric(total_pages),  na.rm=TRUE)

# master vectors to store the data we will scrape
all_listing_names <- c()
all_listing_prices <- c()
all_listing_details <-c()
all_listing_locations <-c()

for(i in 1:total_pages) {
  print(paste('Accessing page:', i))
  # paste the current page number on to the end of the link
  target_url <- paste0("https://www.preloved.co.uk/classifieds/pets/dogs?page=",i)
  target_page <- read_html(target_url)
  
  
  # extract the listing name, price, descriptions and location
  
  listing_title <- target_page |> 
    html_elements('.search-result__content') |> 
    html_elements('.search-result__header') |> 
    html_elements('h2') |> 
    html_elements('.is-title') |> 
    html_text() 
  
  listing_price <- target_page |>  
    html_elements('div.search-result__content') |> 
    html_elements('.search-result__header') |> 
    html_elements('.is-price ') |>  
    html_text() 
  
  listing_location <- target_page |>  
    html_elements('.search-result__content') |> 
    html_elements('.search-result__location') |> 
    html_elements('.is-location') |> 
    html_text() 
  
  listing_details <- target_page |> 
    html_elements('.search-result__content') |> 
    html_elements('.search-result__description') |>  
    html_text() 
  
  # append on to master vectors
  all_listing_names <- append(all_listing_names,listing_title)
  all_listing_prices <- append(all_listing_prices,listing_price)
  all_listing_locations <- append(all_listing_locations,listing_location)
  all_listing_details <- append(all_listing_details,listing_details)
  
  # creating a random sleep time: give me a single number between 1 and 5
  Sys.sleep(runif(1, 1, 5))
  
  print('--- NEXT PAGE ---')
  
  
}

## CLEANING DATA----

#remove whitespace and replace new line in details with a space. 
all_listing_names <- str_trim(all_listing_names)
all_listing_prices <- str_trim(all_listing_prices)

all_listing_details <- str_trim(all_listing_details)
all_listing_details<- gsub("[\n]", " ",all_listing_details)

#remove any additional text from price leaving out the currency sign and price.    
all_listing_prices<-str_extract(all_listing_prices,"[£€$]?[0-9]+(.[0-9]+)?")


# Now that all our infomation is vectorised, we can turn this into a dataframe
final_df <- data.frame("ListingName" = all_listing_names, 
                       "Price" = all_listing_prices,
                       "Details" = all_listing_details,  
                       "Location" = all_listing_locations)

View(final_df)
# Wrie everyting to a .csv file 
write.csv(final_df,file = "dogsforsale.csv",row.names = TRUE)
