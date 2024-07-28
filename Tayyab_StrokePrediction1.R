
##################################################
#Step 1: Data Cleaning
##################################################

library(dplyr)
library(remotes)
library(fastDummies)
library(caret)
library(VIM)
library(ggplot2)
library(ROSE) 
library(lattice)
library(randomForest)
library(pROC)

#loading data from your working directory
strokedata<-read.csv("./strokedata.csv")

#viewing data
View(strokedata)

#examining data
#5110 observations, 12 variables
names(strokedata)
head(strokedata)
summary(strokedata) 

#converting appropriate variables into categorical types
strokedata$gender <- factor(strokedata$gender)
strokedata$ever_married <- factor(strokedata$ever_married)
strokedata$work_type <- factor(strokedata$work_type)
strokedata$Residence_type <- factor(strokedata$Residence_type)
strokedata$smoking_status <- factor(strokedata$smoking_status)
strokedata$stroke <- factor(strokedata$stroke)

#converting bmi into a numerical data type
strokedata$bmi <- as.numeric(strokedata$bmi)

#converting smoking_status unknown category to NA and removing the unknown 
#category
strokedata$smoking_status[strokedata$smoking_status == "Unknown"] <- NA
strokedata$smoking_status <- droplevels(strokedata$smoking_status) 

#dropping id
strokedata <- strokedata %>%
  select(-id)

#### Examining missingness 

library(VIM) #checking missing data
aggr_plot <- aggr(strokedata, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

#~30% of data in smoking status is missing. Best to not impute and drop

strokedata1 <- select(strokedata, -smoking_status)

#Checking if missingness in BMI has a pattern

na.gender <- strokedata %>%
  group_by(gender) %>%
  summarise(na_count = sum(is.na(bmi)), total_count = n(), na_percentage = na_count / total_count * 100, .groups = 'drop')
print(na.gender) #3.24% females, 4.92% males

# Create the bar plot
ggplot(na.gender, aes(x = as.factor(gender), y = na_percentage, fill = gender)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Percentage of Missing BMI Values by Gender",
    x = "Gender",
    y = "Percentage of Missing BMI Values (%)"
  ) +
  theme_minimal() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = sprintf("%.2f%%", na_percentage)), vjust = -0.5)


na.hypertension <- strokedata %>%
  group_by(hypertension) %>%
  summarise(na_count = sum(is.na(bmi)), total_count = n(), na_percentage = na_count / total_count * 100, .groups = 'drop')
print(na.hypertension) #3.34% no hypertension. 9.44% hypertension but ~90% of ppl have no 
#hypertension also so skewed already

# Create the bar plot
ggplot(na.hypertension, aes(x = as.factor(hypertension), y = na_percentage, fill = hypertension)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Percentage of Missing BMI Values by Hypertension",
    x = "Gender",
    y = "Percentage of Missing BMI Values (%)"
  ) +
  theme_minimal() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = sprintf("%.2f%%", na_percentage)), vjust = -0.5)

na.heart_disease <- strokedata %>%
  group_by(heart_disease) %>%
  summarise(na_count = sum(is.na(bmi)), total_count = n(), na_percentage = na_count / total_count * 100, .groups = 'drop')
print(na.heart_disease) #3.4% no hd. 12% hd. >90% of ppl have no hd so more people who had HD didn't report BMI

na.ever_married <- strokedata %>%
  group_by(ever_married) %>%
  summarise(na_count = sum(is.na(bmi)), total_count = n(), na_percentage = na_count / total_count * 100, .groups = 'drop')
print(na.ever_married) 

na.stroke <- strokedata %>%
  group_by(stroke) %>%
  summarise(na_count = sum(is.na(bmi)), total_count = n(), na_percentage = na_count / total_count * 100, .groups = 'drop')
print(na.stroke) #3.31% NAs in no stroke, 16.1% NAs in stroke

#visually, enough to say that data is not MCAR (missing completely at random)
#seems like it is MAR (missing at random)

#replace NAs in BMI with mean BMI
# Calculate the mean of 'var', ignoring NA values
mean_value <- mean(strokedata1$bmi, na.rm = TRUE)

# Replace NA values with the mean
strokedata1$bmi[is.na(strokedata1$bmi)] <- mean_value

#converting appropriate variables to numeric for ML Model predictions
strokedata$gender <- as.numeric(strokedata$gender) 
strokedata$hypertension <- as.numeric(strokedata$hypertension)
strokedata$heart_disease <- as.numeric(strokedata$heart_disease)
strokedata$ever_married <- as.numeric(strokedata$ever_married)

#One-hot encoding 

# 1. One-hot encoding -work_type
work_type_dummy <- dummyVars(" ~ work_type", data = strokedata, sep = "_")
work_type_encoded <- predict(work_type_dummy, strokedata)
# Add the encoded columns to original dataset
strokedata <- cbind(strokedata, work_type_encoded)


# 2. One-hot encoding -residence_type
residence_type_dummy <- dummyVars(" ~ Residence_type", data = strokedata, sep = "_")
residence_type_encoded <- predict(residence_type_dummy, strokedata)
# Add the encoded columns to original dataset
strokedata <- cbind(strokedata, residence_type_encoded)

strokedata <- strokedata%>%
  select(-work_type)
strokedata <- strokedata%>%
  select(-Residence_type)
strokedata <- strokedata %>% 
  rename(work_type_Self = `work_type_Self-employed`)
summary(strokedata)

#Class imbalance
summary(strokedata$stroke)
percentage_data <- strokedata %>%
  group_by(stroke) %>%
  summarise(percentage = n() / nrow(strokedata) * 100)
ggplot(percentage_data, aes(x = stroke, y = percentage, fill = stroke)) +
  geom_bar(stat = "identity") +
  labs(title = "Distribution of Stroke Cases",
       x = "Stroke",
       y = "Percentage (%)") +
  theme_minimal()
#there is a class imabalance, about 95% of the sample did not have a stroke

#Balancing Class Variable (stroke) 
stroke1 <- ovun.sample(stroke~., data=strokedata1,
                                 N=nrow(strokedata1), p=0.5,
                                 seed=1, method="both")$data

#checking if target variable balanced
barplot(table(stroke1$stroke))


##################################################
#Step 2: Building Predictive Models
##################################################

#Splitting data into training and validating

# create a list of 80% of the rows in the dataset we can use for training
validation_index <- createDataPartition(stroke1$stroke, p=0.80, list=FALSE)
# select 20% of the data for validation
validation1 <- stroke1[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset1 <- stroke1[validation_index,]

#normalizing validation and training dataset 

#1
preprocessed_data1 <- preProcess(dataset1, method = 'range')
transformed_data1 <- predict(preprocessed_data1, dataset1)

preprocessed_data1.1 <- preProcess(validation1, method = 'range')
tvalidated_data1 <- predict(preprocessed_data1.1, validation1)

#create models
# Run algorithms using 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"

#Creating Models 
# a) Generalized Linear Model
set.seed(7)
fit.glmnet <- train(stroke~., data=transformed_data1, method="glmnet", metric=metric, trControl=control)
# CART
fit.cart <- train(stroke~., data=transformed_data1, method="rpart", metric=metric, trControl=control)
# b) nonlinear algorithms
# kNN
set.seed(7)
fit.knn <- train(stroke~., data=transformed_data1, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(stroke~., data=transformed_data1, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
library(randomForest)
fit.rf <- train(stroke~., data=transformed_data1, method="rf", metric=metric, trControl=control)


#Summarize accuracy of models 
results <- resamples(list(glmnet=fit.glmnet, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)

#Validation for RF
predictions <- predict(fit.rf, tvalidated_data1)
confusionMatrix(predictions, tvalidated_data1$stroke)

#ROC Curve
predictions <- predict(fit.rf, tvalidated_data1, type="prob")
predicted_probabilities <- predictions[, "1"]
roc_curve <- roc(tvalidated_data1$stroke, predicted_probabilities)
plot(roc_curve, main="ROC Curve for Random Forest Model", legacy.axes=TRUE)


#Feature importance
feature_importance <- varImp(fit.rf, scale = FALSE)$importance

impFeatures_raw <- data.frame(
  Feature = rownames(feature_importance),
  Importance = feature_importance[, "Overall"]  # Adjust column name based on the output
)

# Normalize the importance scores
impFeatures_raw <- impFeatures_raw %>%
  mutate(
    Normalized_Importance = Importance / sum(Importance)  # Normalize to sum to 1
  ) %>%
  arrange(desc(Normalized_Importance))

# Print the normalized feature importance
print(impFeatures_raw)

# Visualizing featured importance 
ggplot(impFeatures_raw, aes(x = reorder(Feature, Normalized_Importance), y = Normalized_Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # Flip coordinates for better readability
  labs(
    x = "Feature",
    y = "Normalized Importance"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    plot.title = element_text(size = 16, face = "bold")
  )

