#This code builds an access pipeline to The Guardian, and identify pattern in retrieved text

#load the library----
library(guardianapi)
library(plyr)
library(tidyverse)
library(lubridate)
library(rvest)

# register and obtain an API key from The Guardian Open Platform for authentication.
# check if an API key is stored in the session variable
gu_api_key()

#build a query
my_query_2023 <- gu_content(query= 'stabbing',
                         from_date="2023-01-01",
                         to_date="2023-12-31")

#determine the dimension of my query
dim(my_query_2023)

#access the headline and body_text of the article in position 1
my_query_2023$headline[1]
my_query_2023$body[1]
my_query_2023$body_text[1]

sectioncounts <- table(my_query_2023$section_id)
barplot(sectioncounts, xlab="sections", ylab="frequency") 

# function to convert table into a data.frame
sections_df <-as.data.frame(sectioncounts) |> 
  filter(Freq > 10) # filter those with over 10 counts

# plot a pie chart on that
pie(sections_df$Freq, sections_df$Var1, main="Sections of The Guardian covering stabbing cases")

#identifting patterns in retrieved text-------------
my_query_2023$word_count_converted <- as.numeric(as.character(my_query_2023$wordcount))

# plot the word count per article for "police" and "trust"
plot(my_query_2023$word_count_converted
     , main='Word count per article for "stabbing"'
     , ylab = 'No. of words')

# assess how many word count values arer larger that the mean plus 3 SD
mean3sd <- mean(my_query_2023$word_count_converted) + 3*sd(my_query_2023$word_count_converted)
table(my_query_2023$word_count_converted > mean3sd)

# Looking for the temporal development of article length
# extracts the first 10 characters of the webPublicationDate string
my_query_2023$date <- substr(my_query_2023$web_publication_date, 1, 10)

# convert the new variable to a date format variable
my_query_2023$date <- as.Date(my_query_2023$date, format = "%Y-%m-%d")

# sort the data frame by date
my_query_2023 <- my_query_2023 |> 
  arrange(date)

# Plot the word count over time
plot(x = my_query_2023$date
     , y = my_query_2023$word_count_converted
     , type="p"
     , main = 'Word count for articles with keyword "stabbing" over time')

# assess how the article frequency changes over time
# create a month variable based on the date variable
my_query_2023$month <- month(my_query_2023$date)
# create a list of months
monthslist <- list(months=my_query_2023$month)
#aggregate the data by month
by_month <- aggregate(my_query_2023$word_count_converted
                      , by = monthslist
                      , FUN = sum)

plot(by_month$months
     , y = by_month$x
     , type="b"
     , ylab = 'Sum of words'
     , xlab = 'Month'
     , main = 'Sum of words per month')
