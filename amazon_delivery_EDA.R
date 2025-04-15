install.packages("geosphere")
install.packages("hms")
install.packages("dplyr")
install.packages("fastDummies")
install.packages("caret")

library(geosphere)
library(hms)
library(dplyr)
library(data.table)
library(fastDummies)
library(caret)

amazon_data <- read.csv('amazon_delivery.csv', na.strings = c("NA",""))
str(amazon_data)
summary(amazon_data)

nrow(amazon_data)

# converting all NaN characters to NA
amazon_data <- lapply(amazon_data, function(x) {
  x[x == "NaN" | x == "NaN "] <- NA
  return(x)
})
amazon_data <- as.data.frame(amazon_data)
amazon_data_na <- apply(is.na(amazon_data), 2, sum)
amazon_data_na


amazon_new <- na.omit(amazon_data)
nrow(amazon_new)

# removing outliers in target variable
Q1 <- quantile(amazon_new[["Delivery_Time"]], 0.25)
Q3 <- quantile(amazon_new[["Delivery_Time"]], 0.75)
IQR <- Q3 - Q1
LB <- Q1 - 1.5 * IQR
UB <- Q3 + 1.5 * IQR
amazon_new_o <- amazon_new[amazon_new[["Delivery_Time"]] >= LB & amazon_new[["Delivery_Time"]] <= UB, ]

nrow(amazon_new_o)

amazon_clean <- unique(amazon_new_o)
nrow(amazon_clean)

amazon_clean$distance_meters <- distHaversine(
  matrix(c(amazon_clean$Store_Longitude, amazon_clean$Store_Latitude), ncol = 2),
  matrix(c(amazon_clean$Drop_Longitude, amazon_clean$Drop_Latitude), ncol = 2)
)

amazon_clean$distance_km <- amazon_clean$distance_meters / 1000




amazon_clean$Order_Date <- as.Date(amazon_clean$Order_Date, format = "%Y-%m-%d")

amazon_clean$order_day <- as.integer(format(amazon_clean$Order_Date, "%d"))
amazon_clean$order_month <- as.integer(format(amazon_clean$Order_Date, "%m"))
amazon_clean$order_year <- as.integer(format(amazon_clean$Order_Date, "%Y"))
amazon_clean$order_day_of_week <- as.integer(format(amazon_clean$Order_Date, "%u"))

amazon_clean$Order_Time <- strptime(amazon_clean$Order_Time, format = "%H:%M:%S")

amazon_clean$Pickup_Time <- strptime(amazon_clean$Pickup_Time, format = "%H:%M:%S")



amazon_clean$order_hour <- as.numeric(format(amazon_clean$Order_Time, "%H"))
amazon_clean$order_min <- as.numeric(format(amazon_clean$Order_Time, "%M"))
amazon_clean$order_sec <- as.numeric(format(amazon_clean$Order_Time, "%S"))

amazon_clean$order_time_dec <- amazon_clean$order_hour + amazon_clean$order_min/60 + amazon_clean$order_sec/3600

amazon_clean$pickup_hour <- as.numeric(format(amazon_clean$Pickup_Time, "%H"))
amazon_clean$pickup_min <- as.numeric(format(amazon_clean$Pickup_Time, "%M"))
amazon_clean$pickup_sec <- as.numeric(format(amazon_clean$Pickup_Time, "%S"))

amazon_clean$pickup_time_dec <- amazon_clean$pickup_hour + amazon_clean$pickup_min/60 + amazon_clean$pickup_sec / 3600

amazon_clean$order_to_pickup_duration <- amazon_clean$pickup_time_dec - amazon_clean$order_time_dec

amazon_clean$order_time_dec_sin <- sin(2 * pi * amazon_clean$order_time_dec / 24)
amazon_clean$order_time_dec_cos <- cos(2 * pi * amazon_clean$order_time_dec / 24)

amazon_clean$pickup_hour_sin <- sin(2 * pi * amazon_clean$pickup_time_dec / 24)
amazon_clean$pickup_hour_cos <- cos(2 * pi * amazon_clean$pickup_time_dec / 24)

amazon_clean <- amazon_clean %>% select(-Store_Latitude, -Store_Longitude, -Drop_Latitude, -Drop_Longitude, -Order_Date, -Order_Time, -Pickup_Time,-distance_meters, -order_hour, -order_min, -order_sec, -pickup_hour, -pickup_min, -pickup_sec)

data_encoded <- dummy_cols(amazon_clean, select_columns = c("Weather", "Traffic", "Vehicle", "Area", "Category"), remove_first_dummy = TRUE)
data_encoded <- data_encoded %>% select(-Order_ID,-Weather, -Traffic, -Vehicle, -Area, -Category,  -order_time_dec_sin, -order_time_dec_cos, -pickup_hour_sin, -pickup_hour_cos)

min_max_scale <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

data_scaled <- data.frame(apply(data_encoded,2, function(x) if(is.numeric(x)) min_max_scale(x) else x))

set.seed(123)

# Create a 70-15-15 split: 70% train, 15% test, 15% validation
train_index <- createDataPartition(data_scaled$Delivery_Time, p = 0.7, list = FALSE)
train_data <- data_scaled[train_index, ]
remaining_data <- data_scaled[-train_index, ]

# Split remaining data into test and validation (50% each of the 30% remaining)
test_index <- createDataPartition(remaining_data$Delivery_Time, p = 0.5, list = FALSE)
test_data <- remaining_data[test_index, ]
validation_data <- remaining_data[-test_index, ]

# Check the sizes of each dataset
cat("Training Data Size: ", nrow(train_data), "\n")
cat("Test Data Size: ", nrow(test_data), "\n")
cat("Validation Data Size: ", nrow(validation_data), "\n")

write.csv(train_data,"train.csv")
write.csv(test_data,"test.csv")
write.csv(validation_data,"validation.csv")
