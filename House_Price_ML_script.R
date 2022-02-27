
#  Dataset  <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>
#  House Prices Prediction - Kaggle dataset 
#  Install and load the packages below :

if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(dummies)) install.packages("dummies", repos = "http://cran.us.r-project.org")
if(!require(lares)) install.packages("lattice", repos = "http://cran.us.r-project.org")


library(plyr)
library(tidyverse)
library(Matrix)
library(glmnet)
library(lattice)
library(caret)
library(dummies)
library(e1071)
library(lares)


# 1. Data import and initial exploration:--------------------------------------------------------

# the dataset is offered in two separated files, one for the training and another one for
# test set. 
# dataset is available on this web-page:  <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>

training_data = read.csv(file = file.path("https://raw.githubusercontent.com/kkostanjevec/HarvardCap_House_price_pred/main/train.csv"))
test_data = read.csv(file = file.path("https://raw.githubusercontent.com/kkostanjevec/HarvardCap_House_price_pred/main/test.csv"))

str(training_data)
str(test_data)  
# ATTENTION: it can be seen that test_data (test.csv) HAS NO SalePrice variable 
# (it has 1 variable less than train.csv). 
# Therefore, the only useful df for ML training, testing and validating is training data.

# join datasets for data cleaning 
test_data$SalePrice <- 0
dataset <- rbind(training_data, test_data)


# 2. Data cleaning and dealing with NAs ---------------------------------------------------------

# data set is filled with missing values - this needs to be addressed
na.cols <- which(colSums(is.na(dataset)) > 0)
sort(colSums(sapply(dataset[na.cols], is.na)), decreasing = TRUE)
paste('There are', length(na.cols), 'columns with missing values')

# dealing with numerical variable - assume that `NAs` in these variables means 0.
# e.g. LotFrontage : NA most likely means no lot frontage
dataset$LotFrontage[is.na(dataset$LotFrontage)] <- 0
dataset$MasVnrArea[is.na(dataset$MasVnrArea)] <- 0
dataset$BsmtFinSF1[is.na(dataset$BsmtFinSF1)] <- 0
dataset$BsmtFinSF2[is.na(dataset$BsmtFinSF2)] <- 0
dataset$BsmtUnfSF[is.na(dataset$BsmtUnfSF)] <- 0
dataset$TotalBsmtSF[is.na(dataset$TotalBsmtSF)] <- 0
dataset$BsmtFullBath[is.na(dataset$BsmtFullBath)] <- 0
dataset$BsmtHalfBath[is.na(dataset$BsmtHalfBath)] <- 0
dataset$GarageCars[is.na(dataset$GarageCars)] <- 0
dataset$GarageArea[is.na(dataset$GarageArea)] <- 0

# for the variable "GarageYrBlt". We can assume that the year the garage was built is the same when the house itself was built.
dataset$GarageYrBlt[is.na(dataset$GarageYrBlt)] <- dataset$YearBuilt[is.na(dataset$GarageYrBlt)]
summary(dataset$GarageYrBlt)
# correcting the error in the dataset
dataset$GarageYrBlt[dataset$GarageYrBlt==2207] <- 2007

# dealing with `NAs` in categorical values.
# we find "real" NAs, then impute them with the most common value for this feature.
dataset$KitchenQual[is.na(dataset$KitchenQual)] <- names(sort(-table(dataset$KitchenQual)))[1]
dataset$MSZoning[is.na(dataset$MSZoning)] <- names(sort(-table(dataset$MSZoning)))[1]
dataset$SaleType[is.na(dataset$SaleType)] <- names(sort(-table(dataset$SaleType)))[1]
dataset$Exterior1st[is.na(dataset$Exterior1st)] <- names(sort(-table(dataset$Exterior1st)))[1]
dataset$Exterior2nd[is.na(dataset$Exterior2nd)] <- names(sort(-table(dataset$Exterior2nd)))[1]
dataset$Functional[is.na(dataset$Functional)] <- names(sort(-table(dataset$Functional)))[1]

# for empty values, we just change the `NA` value to a new value - 'No', for the rest we change NAs to their actual meaning
# e.g. NA for basement features is "no basement", etc.
dataset$Alley = factor(dataset$Alley, levels=c(levels(dataset$Alley), "No"))
dataset$Alley[is.na(dataset$Alley)] = "No"
dataset$BsmtQual = factor(dataset$BsmtQual, levels=c(levels(dataset$BsmtQual), "No"))
dataset$BsmtQual[is.na(dataset$BsmtQual)] = "No"
dataset$BsmtCond = factor(dataset$BsmtCond, levels=c(levels(dataset$BsmtCond), "No"))
dataset$BsmtCond[is.na(dataset$BsmtCond)] = "No"
dataset$BsmtExposure[is.na(dataset$BsmtExposure)] = "No"
dataset$BsmtFinType1 = factor(dataset$BsmtFinType1, levels=c(levels(dataset$BsmtFinType1), "No"))
dataset$BsmtFinType1[is.na(dataset$BsmtFinType1)] = "No"
dataset$BsmtFinType2 = factor(dataset$BsmtFinType2, levels=c(levels(dataset$BsmtFinType2), "No"))
dataset$BsmtFinType2[is.na(dataset$BsmtFinType2)] = "No"
dataset$Fence = factor(dataset$Fence, levels=c(levels(dataset$Fence), "No"))
dataset$Fence[is.na(dataset$Fence)] = "No"
dataset$FireplaceQu = factor(dataset$FireplaceQu, levels=c(levels(dataset$FireplaceQu), "No"))
dataset$FireplaceQu[is.na(dataset$FireplaceQu)] = "No"
dataset$GarageType = factor(dataset$GarageType, levels=c(levels(dataset$GarageType), "No"))
dataset$GarageType[is.na(dataset$GarageType)] = "No"
dataset$GarageFinish = factor(dataset$GarageFinish, levels=c(levels(dataset$GarageFinish), "No"))
dataset$GarageFinish[is.na(dataset$GarageFinish)] = "No"
dataset$GarageQual = factor(dataset$GarageQual, levels=c(levels(dataset$GarageQual), "No"))
dataset$GarageQual[is.na(dataset$GarageQual)] = "No"
dataset$GarageCond = factor(dataset$GarageCond, levels=c(levels(dataset$GarageCond), "No"))
dataset$GarageCond[is.na(dataset$GarageCond)] = "No"
dataset$MasVnrType = factor(dataset$MasVnrType, levels=c(levels(dataset$MasVnrType), "No"))
dataset$MasVnrType[is.na(dataset$MasVnrType)] = "No"
dataset$MiscFeature = factor(dataset$MiscFeature, levels=c(levels(dataset$MiscFeature), "No"))
dataset$MiscFeature[is.na(dataset$MiscFeature)] = "No"
dataset$PoolQC = factor(dataset$PoolQC, levels=c(levels(dataset$PoolQC), "No"))
dataset$PoolQC[is.na(dataset$PoolQC)] = "No"
dataset$Electrical = factor(dataset$Electrical, levels=c(levels(dataset$Electrical), "UNK"))
dataset$Electrical[is.na(dataset$Electrical)] = "UNK"

# remove some unnecessary features 
dataset$Utilities <- NULL
dataset$Id <- NULL

# now check again if we have null values.
na.cols <- which(colSums(is.na(dataset)) > 0)
paste('There are now', length(na.cols), 'columns with missing values')


# 3. Data engineering and remodeling -----------------------------------------------------------

## Recoding descriptive variables into ordinal
dataset$ExterQual<- recode(dataset$ExterQual,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$ExterCond<- recode(dataset$ExterCond,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$BsmtQual<- recode(dataset$BsmtQual,"No"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$BsmtCond<- recode(dataset$BsmtCond,"No"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$BsmtExposure<- recode(dataset$BsmtExposure,"No"=0,"No"=1,"Mn"=2,"Av"=3,"Gd"=6)
dataset$BsmtFinType1<- recode(dataset$BsmtFinType1,"No"=0,"Unf"=1,"LwQ"=2,"Rec"=3,"BLQ"=4,"ALQ"=5,"GLQ"=6)
dataset$BsmtFinType2<- recode(dataset$BsmtFinType2,"No"=0,"Unf"=1,"LwQ"=2,"Rec"=3,"BLQ"=4,"ALQ"=5,"GLQ"=6)
dataset$HeatingQC<- recode(dataset$HeatingQC,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$KitchenQual<- recode(dataset$KitchenQual,"None"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$Functional<- recode(dataset$Functional,"None"=0,"Sev"=1,"Maj2"=2,"Maj1"=3,"Mod"=4,"Min2"=5,"Min1"=6,"Typ"=7)
dataset$FireplaceQu<- recode(dataset$FireplaceQu,"No"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$GarageFinish<- recode(dataset$GarageFinish,"No"=0,"Unf"=1,"RFn"=2,"Fin"=3)
dataset$GarageQual<- recode(dataset$GarageQual,"No"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$GarageCond<- recode(dataset$GarageCond,"No"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$PoolQC<- recode(dataset$PoolQC,"No"=0,"Po"=1,"Fa"=2,"TA"=3,"Gd"=4,"Ex"=6)
dataset$Fence<- recode(dataset$Fence,"No"=0,"MnWw"=1,"GdWo"=2,"MnPrv"=3,"GdPrv"=6)

## Adding the new features:
# i) new feature - area/size
# Total surface of the house, combining the total inside surface (1 and 2 floor square feet)
dataset['TotalInsideSF'] <- as.numeric(dataset$X1stFlrSF + dataset$X2ndFlrSF)

# ii) new feature - quality of the foundation material of a house
#     a numerical feature to codify the ranking of the foundation material according to the median house value.
training_data[,c('Foundation','SalePrice')] %>%
  group_by(Foundation) %>%
  summarise(avg = median(SalePrice, na.rm = TRUE)) %>%
  arrange(avg) %>%
  mutate(sorted = factor(Foundation, levels=Foundation)) %>%
  ggplot(aes(x=sorted, y=avg)) +
  geom_bar(stat = "identity", fill="grey") + 
  scale_y_continuous(labels = scales::comma)+
  labs(x='Foundation', y='Price in $') +
  theme_minimal()+
  theme(axis.text.x = element_text(angle=45))+
  labs(title ="House foundation - median prices",
       subtitle="in order to codify the ranking of foundation types according to their median price")

dataset$FoundationScore <- recode(dataset$Foundation, 'Slab' = 1, 'BrkTil' = 2, 'Stone' = 2, 'CBlock' = 3, 'Wood' = 4, 'PConc' = 6) 

# iii) new feature - location
# It is common wisdom that the location of a house (neighborhood) is one of the most important 
# categorical predictor of its price. Therefore, the new features is created:
# a numerical feature to codify the ranking of the neighborhoods according to their median house value.

training_data[,c('Neighborhood','SalePrice')] %>%
  group_by(Neighborhood) %>%
  summarise(avg = median(SalePrice, na.rm = TRUE)) %>%
  arrange(avg) %>%
  mutate(sorted = factor(Neighborhood, levels=Neighborhood)) %>%
  ggplot(aes(x=sorted, y=avg)) +
  geom_bar(stat = "identity", fill="grey") + 
  scale_y_continuous(labels = scales::comma)+
  labs(x='Neighborhood', y='Price in $') +
  theme_minimal()+
  theme(axis.text.x = element_text(angle=45))+
  labs(title ="Median prices in various neighborhoods",
       subtitle="in order to codify the ranking of the neighborhoods according to their median price")

dataset$NeighborhoodScored <- recode(dataset$Neighborhood, 'MeadowV' = 1, 'IDOTRR' = 2, 'BrDale' = 2, 'OldTown' = 3, 'Edwards' = 3, 'BrkSide' = 3,'Sawyer' = 4, 'Blueste' = 4, 'SWISU' = 4, 'NAmes' = 4, 'NPkVill' = 4, 'Mitchel' = 4,'SawyerW' = 5, 'Gilbert' = 5, 'NWAmes' =5, 'Blmngtn' = 5, 'CollgCr' = 5, 'ClearCr' = 5, 'Crawfor' = 5, 'Veenker' = 6, 'Somerst' = 6, 'Timber' = 6, 'StoneBr' = 7, 'NoRidge' = 7, 'NridgHt' =7)

## Dealing with Skewness - Transform the target value - SalePrice - applying log
dataset$SalePrice <- log(dataset$SalePrice)

## Factorize features
dataset$MSSubClass <- as.factor(dataset$MSSubClass)
dataset$MoSold <- as.factor(dataset$MoSold)
dataset$YrSold <- as.factor(dataset$YrSold)


# 4. Data visualizations and correlations --------------------------------------------------------

# ATTENTION: as stated at the beginning:
# if we check the test_data from Kaggle we can see it HAS NO SalePrice variable
# Therefore, we will need to focus on the training part of the data-set (i.e. now cleaned dataset1[1:1460,]) 
# for training, testing and validating ML models.

train <- dataset[1:1460,]
summary(train) # check the statistical summaries of the cleaned data-set

# Correlations of variables with Sale Price - in engineered, cleaned and remodeled data-set  
# In order to see which variables are the most important for predicting the sale price of 
# a houses we will check correlations:

corr_var(train,                # name of dataset (which is now cleaned and remodeled)
         SalePrice,            # name of variable to focus on
         method = "pearson",   # name of correlation approach
         top = 20)             # display top 25 correlations

# The graph above helps us to evaluate the dependent variables that are most important in
# predicting the SalePrice variable of houses.
# We can see from the correlations that the most important aspects of a house for predicting 
# its sale price can be summarized in 4 dimensions: 
# 1. Dimension - Quality - includes variables: 
#      OverallQual (Rates the overall material and finish of the house - most correlated
#      KitchenQual (Kitchen quality); 
#      FullBath (Full bathrooms above ground); 
#      FoundationScore(Type of foundation - score on the 1-6 score-list) 
#      Fireplaces: Number of fireplaces; 
#      HeatingQC: Heating quality and condition; 
#      ExterQual (Evaluates the quality of the material on the exterior)

# 2. Dimension - Location - includes variable 
#      NeighborhoodScored (location of a house on the 1-7 score-list)

# 3. Dimension - Size - includes variables: 
#      TotalInsideSF (Total inside surface of a house, combining 1st and 2nd floor); 
#      GarageCars (Size of garage in car capacity); 
#      GrLivArea: (Above ground living area square feet); 
#      GarageArea (Size of garage in square feet); 
#      TotalBsmtSF (Total square feet of basement area); 
#      TotRmsAbvGrd (Total rooms above grade) 

# 4. Dimension - Age - includes variables: 
#     YearBuilt (Original construction date); 
#     YearRemodAdd (Remodel of a house date)

# We can visualize the most important aspects of these 4 dimensions and
# put them in relation with SalePrice variable - the variable why are trying to predict 

# Exploring the sale price variable (the outcome we want to predict with ML model)
ggplot(training_data, aes(SalePrice)) +
  geom_histogram(fill="grey", color="blue") +
  scale_x_continuous(labels = scales::comma)+
  labs(title="Distribution of Sale Prices",
       subtitle = "Original prices (without log) from the dataset",
       x="Sale Prices in $")+
  theme_bw()

ggplot(train, aes(SalePrice)) +
  geom_histogram(fill="grey", color="blue") +
  labs(title="Distribution of Sale Prices",
       subtitle="Log transformed prices - which will be used for ML models",
       x="Sale Prices in log")+
  theme_bw()

# The distribution shows that most of the houses are sold under $200,000. 
# This is confirmed if we statistically summarize Sale Price variable: 
# mean price is 180.000$ and median is 163.000$.
# The SalePrice in the dataset is right skewed:  most expensive houses greater than 400.000$ 
# have lower volume of sales as compared with the majority of houses in the 100.000-300.000$ range.

summary(training_data$SalePrice) # original $ prices
summary(train$SalePrice)         # log prices

# We have seen from correlations that 4 most important aspects of the house that determine 
# its price are its quality, neighborhood, size and age.
# It will be useful to visualize the most important aspects of these 4 dimensions 
# and put it in a relation with SalePrice variable 
# (*although later for ML models we will use log transformed Sale Price variable, now
# for some visualizations we will use training_data df with the original Sale Price 
# values because it is easier to interpret original values than log transformed ones):

# Quality (1. dimension) and Age (4. dimension) of a house vs. its Sale Price 
training_data %>%
  ggplot(aes(factor(OverallQual), SalePrice))+
  geom_point(alpha = 1, aes(color = YearBuilt))+
  geom_boxplot(alpha =0.01, aes(group=OverallQual))+
  geom_hline (aes(yintercept = mean(SalePrice)), color="red")+
  theme_minimal()+
  scale_y_continuous(labels = scales::comma)+
  labs(title = "Quality and age of a house vs its Sale Price",
       subtitle = "In general, houses with higher quality-grade are more expensive and newer (red line = mean price)",
       x= "Overall Quality Grade (from 1 to 10)",
       y= "Sell Price in $")

# Location (2. dimension) and Age (4. dimension) of a house vs. its Sale Price 
training_data %>% 
  ggplot(aes(reorder(Neighborhood, SalePrice), SalePrice)) +
  geom_point(alpha = 1, aes(color = YearBuilt))+
  geom_boxplot(alpha =0.01) +
  geom_hline(aes(yintercept = mean(SalePrice)),color="red") +
  theme_minimal()+
  scale_y_continuous(labels = scales::comma)+
  theme(axis.text.x = element_text(angle = 40, hjust = 1)) +
  labs(title = "Location of a house vs its Sale Price",
       subtitle = "In general, newer houses are located in more expansive neighborhoods (red line = mean price) ",
       x= "Location - Neighborhood",
       y= "Sell Price in $")

# Size (3. dimension) and Foundation Quality (1. dimension)  of a house vs. its Sale Price 
# size showed with GrLivArea variable = Above grade (ground) living area square fee; 
# foundation quality showed with our custom made variable (1-6 scale)
# sale price on a log scale which will be used for ML models
train %>%
  ggplot(aes(GrLivArea, SalePrice))+
  geom_point(alpha = 0.5, aes(color = FoundationScore))+
  geom_smooth(method = "lm")+
  theme_minimal()+
  labs(title = "Size of a house vs its Sale Price",
       subtitle = "Larger houses are more expansive, with foundation quality visibly influencing the price",
       x= "Above ground living area square feet",
       y= "Sell Price in log",
       color="Foundation quality:\n6=best (concrete) \n1=worst(slab)")

# Size (3. dimension) and Location (2. dimension) of a house vs. its Sale Price 
# size showed with our custom made TotalInsideSF variable (1.+ 2. floor square feet)
# location showed with our custom made NeighborhoodScored variable (1-7 scale) 
# sale Price on log scale which will be used for ML models
train %>%
  ggplot(aes(TotalInsideSF, SalePrice))+
  geom_point(alpha = 0.5, aes(color = NeighborhoodScored))+
  geom_smooth(method = "lm")+
  theme_minimal()+
  labs(title = "Size of a house vs its Sale Price",
       subtitle = "Larger houses are more expansive, with location visibly influencing the price",
       x= "First and second floor square feet",
       y= "Sell Price in log",
       color="Location Score:\n7 = most elite \n1= least elite")


# 5. Machine learning MODELS: -----------------------------------------------------------------
   # a) Data partition: Train set, Test set and Validation set---------------------------------

# For data cleaning and feature engineering, we merged train and test data-sets from Kaggle. 
# We find out that Kaggle test data-set has no  SalePrice variable, so it cannot be useful 
# for using it for testing or validation.
# We will need to focus on the cleaned "train" part of the df (i.e. dataset1[1:1460,]) 
# and split it in 3 parts: 60% of it for train_set, 20% of for test_set and
# 20% of for the final validation. We could enlarge the train part of the data-set and this would
# give us slightly better RMSE results, but this would leave too small proportion of the data 
# for test and validation data-sets. That is why the ratio 60/20/20 seems appropriate.
# Our ML models will focus on previously emphasized 4 dimension variables that we found 
# are the most correlated with the sale price of houses, namely: size, location, age and quality.

# Therefore, we will pick the most correlated variables - with correlation above 0.45 - 
# with the sale price for the subset of the data in order to do ML predictions of the house prices.

# Separation of the top correlated variables with Sale Price (with correlation above 0.45 
# from 4 dimension important for the sale price: size, location, age and quality)
basedf<- subset(train, select = c(OverallQual , NeighborhoodScored, TotalInsideSF, GarageCars, GrLivArea, KitchenQual, GarageArea, TotalBsmtSF , FullBath, YearBuilt, YearRemodAdd, FoundationScore, TotRmsAbvGrd, Fireplaces, HeatingQC, ExterQual, SalePrice))

## Tripartite Data partition: 
set.seed(99, sample.kind = "Rounding")

# Set the fractions of the df for training, validation, and test.
fractionTraining <- 0.6
fractionValidation <- 0.2
fractionTest <- 0.2
# Compute sample sizes.
sampleSizeTraining <- floor(fractionTraining * nrow(basedf))
sampleSizeValidation <- floor(fractionValidation * nrow(basedf))
sampleSizeTest <- floor(fractionTest * nrow(basedf))
# Creating the randomly-sampled indices for the dataframe. Use setdiff() to
# avoid overlapping subsets of indices.
indicesTraining <- sort(sample(seq_len(nrow(basedf)), size=sampleSizeTraining))
indicesNotTraining <- setdiff(seq_len(nrow(basedf)), indicesTraining)
indicesValidation <- sort(sample(indicesNotTraining, size=sampleSizeValidation))
indicesTest <- setdiff(indicesNotTraining, indicesValidation)
# Finally, output the three df for training, test and final validation.
train_set <- basedf[indicesTraining, ]
validation <- basedf[indicesValidation, ]
test_set <- basedf[indicesTest, ]

   # b) ML models: Linear Regression, Ridge Regression, Lasso Regression and Random Forrest ---------------

# Cross-validation plan: Cross-validation is also known as a resampling method because it 
# involves fitting the same statistical method multiple times using different subsets of the 
# data. Here we do ten-fold cross-validation to train our models with the caret package. 
# The function trainControl generates parameters that control how models are created, 
# and here we choose 10 fold "cv" method.
cv_plan <- trainControl(method = "cv", number = 10)

#### ML Model 1: Linear Model
# We will add previously defined cross-validation to our lm model, and we will also do 
# preprocess in order to perform a Principal Component Analysis, and also center and scale 
# predictors and identify predictors with near zero variance.
set.seed(99, sample.kind = "Rounding")

model_lm <- train(SalePrice ~ .,
                  method = "lm",
                  trControl = cv_plan,
                  preProcess = c("nzv", "center", "scale", "pca"),
                  data = train_set)

pred_lm <- predict(model_lm, test_set)
rmse_lm <- RMSE((test_set$SalePrice), pred_lm)
rmse_lm


# ML: GLMnet package
# GLMnet package offers amended regression approach similar to linear regression which 
# also provides a way how to, on one side , penalizes number of non-zero-coefficients 
# - what is then called "lasso regression"; while on the other side, it provides a way how to
# penalizes absolute magnitude of coefficients - what is then called "ridge regression". 
# This helps in dealing with collinearity and small datasets.
# Function tuneGrid offers a way how to choose between pure "ridge regression" 
# (setting the alpha = 0) and pure "lasso regression" (setting the alpha = 1). 
# Other tuning settings are similar to linear regression (except not having PCA).
# We will try both "Ridge regression" and "Lasso regression" approaches

#### ML Model 2.: Ridge regression
set.seed(99, sample.kind = "Rounding")

model_ridge <- train(SalePrice ~ .,
                      data=train_set,
                      tuneGrid = expand.grid(alpha = 0,
                                             lambda = seq(0.0001, 1, length = 20)),
                      method = "glmnet",
                      trControl = cv_plan,
                      preProcess = c("nzv", "center", "scale"))

pred_ridge <- predict(model_ridge, test_set)
rmse_ridge <- RMSE((test_set$SalePrice), pred_ridge)
rmse_ridge

#### ML Model 3.: Lasso regression
set.seed(99, sample.kind = "Rounding")

model_lasso <- train(SalePrice ~ .,
                     data=train_set,
                     tuneGrid = expand.grid(alpha = 1,
                                            lambda = seq(0.0001, 1, length = 20)),
                     method = "glmnet",
                     trControl = cv_plan,
                     preProcess = c("nzv", "center", "scale"))

pred_lasso <- predict(model_lasso, test_set)
rmse_lasso <- RMSE((test_set$SalePrice), pred_lasso)
rmse_lasso

#### ML Model 4: Random Forest 
# In the fourth model we will do a random forest ML approach. For this we will use 
# method = "ranger" from the caret package. Ranger of caret package is a fast implementation 
# of random forest, particularly suited for high dimensional data and for our case of not 
# very high computational power.
# Here most important tuning parameter is the number of randomly selected variables at each 
# split for which we use tuneLength control in the code. The default of tuneLength is 3 
# (it means that it tries 3 different models), but we will set it to 13. 

set.seed(99, sample.kind = "Rounding")

model_rf <- train(SalePrice ~ .,
                  tuneLength = 13,
                  data = train_set,
                  method = "ranger",
                  trControl = cv_plan)

pred_rf <- predict(model_rf, test_set)
rmse_rf <- RMSE((test_set$SalePrice), pred_rf)
rmse_rf
# RMSE score compare
data.frame(Models = c("Linear Reggresion","Ridge Regression", "Lasso Regression", "Random Forest"),
           Train_RMSE = round(c(rmse_lm,rmse_ridge,rmse_lasso,rmse_rf), 6))



   # c) ML: Validation -----------------------
# At the begging of this section we kept 20% of the data for the final validation of our 
# ML models. We will do validation for all results above, in order to
# compare test and validation results of all four ML models.

# Predictions of the Linear model - final validation
pred_val_lm <- predict(model_lm, validation)
rmse_val_lm <- RMSE((validation$SalePrice), pred_val_lm)

## Predictions of Ridge model - final validation
pred_val_ridge <- predict(model_ridge, validation)
rmse_val_ridge <- RMSE((validation$SalePrice), pred_val_ridge)

## Predictions of Lasso model - final validation
pred_val_lasso <- predict(model_lasso, validation)
rmse_val_lasso <- RMSE((validation$SalePrice), pred_val_lasso)

## Predicting 'SalePrice' with the Random Forest model - final validation
pred_val_rf <- predict(model_rf, validation)
rmse_val_rf <- RMSE((validation$SalePrice), pred_val_rf)



# Comparison of RMSEs between train/test and validation sets
data.frame(Model_type = c("Linear Reggresion","Ridge Regression", "Lasso Regression", "Random Forest"),
           RMSE_original_train = c(rmse_lm,rmse_ridge,rmse_lasso,rmse_rf),
           RMSE_validation = c(rmse_val_lm, rmse_val_ridge, rmse_val_lasso, rmse_val_rf))

# It is visible that the final validation-results of our ML models are similar and constant
# with the results we acquired during training. Differences between RMSE values of original 
# train and validation phase are not large. 
# The best result during training (0.144) and final validation (0.135) was the one acquired
# with the Random Forest ML approach . 

# Rmarkdown
# save-image work-space:
save.image (file = "my_work_space.RData")
