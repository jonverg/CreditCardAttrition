#---------------------------------
# DATASET : BANK
#---------------------------------
library(tidyverse)
library(rsample)
library(glmnet)
library(glmnetUtils)
library(forcats)

bank <- read.csv("datasets/BankChurners.csv")
bank %>% head()
bank %>% summary()

bank_clean <- bank %>% 
  mutate(Gender = factor(Gender),
         Income_Category = factor(Income_Category),
         Attrition_Flag = ifelse(Attrition_Flag == "Attrited Customer", 1, 0),
         Attrition_Flag = factor(Attrition_Flag)) %>% 
  filter(Income_Category != "Unknown") %>% 
  select(-c(CLIENTNUM, Dependent_count, Card_Category, Total_Relationship_Count, Contacts_Count_12_mon, Total_Amt_Chng_Q4_Q1,
            Total_Ct_Chng_Q4_Q1,
            Marital_Status,
            Education_Level,
            Avg_Open_To_Buy,
            Avg_Utilization_Ratio,
            Total_Trans_Amt,
            Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1,
            Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2)) 

bank_clean %>% glimpse()

#LOG Transformation
bank_clean$log_Credit_Limit <- log(bank_clean$Credit_Limit)
bank_clean %>% glimpse()

ggplot(bank_clean, aes(x = log_Credit_Limit)) + 
  geom_histogram(bins = 30, fill = 'blue') + 
  labs(title = "Histogram of Credit Limit",
       x = "Log(Credit Limit)",
       y = "Frequency")

summary(bank_clean$log_Credit_Limit)

#TRAIN TEST SPLIT 80/20
set.seed(310)
bank_split <- initial_split(bank_clean, prop = 0.80)
bank_train <- training(bank_split)
bank_test <- testing(bank_split)

#Decision Tree: Classification
library(tree)

tree_bank <- tree(Attrition_Flag ~ .,
                     data = bank_train,
                     control = tree.control(nobs = nrow(bank_train),
                                            mindev = 0.005))

plot(tree_bank)
text(tree_bank, pretty = 0, cex=0.5) 

predicted_bank_test <- predict(tree_bank, newdata = bank_test, type = 'class')
table(predicted_bank_test, bank_test$Attrition_Flag)

cv.bank <- cv.tree(tree_bank, FUN=prune.misclass)
# report the results
print(cv.bank)
#4 leaf nodes is the best number 
plot(cv.bank$size, cv.bank$dev, type="b")

bank_prunedtree <-prune.misclass(tree_bank, best = 4)
plot(bank_prunedtree)
text(bank_prunedtree)

#TRAIN
predicted_bank_train <- predict(bank_prunedtree, newdata = bank_train, type = 'class')
table(predicted_bank_train, bank_train$Attrition_Flag)

#TEST
predicted_bank_prunedtree <- predict(bank_prunedtree, newdata = bank_test, type = 'class')
conf_matrix <- table(predicted_bank_prunedtree, bank_test$Attrition_Flag)
conf_matrix


#TRAIN -- Confusion Matrix:
TN = 5885
TP = 555
FP = 162
FN = 610

#accuracy = 
(TN + TP)/ (TN + TP + FP + FN)
#0.8929 Accuracy
#--
#sensitivity = (TP) / (TP + FN)
(TP) / (TP + FN)
#Sensitivity = 0.476
#--
#specificity = (TN) / (TN + FP)
(TN) / (TN + FP)
#Specificity - 0.973
#--


#TEST -- Confusion Matrix:
TN = 1493
TP = 137
FP = 35
FN = 137

#accuracy = 
(TN + TP)/ (TN + TP + FP + FN)
#0.9045 Accuracy
#--
#sensitivity = (TP) / (TP + FN)
(TP) / (TP + FN)
#Sensitivity = 0.5
#--
#specificity = (TN) / (TN + FP)
(TN) / (TN + FP)
#Specificity - 0.977
#--




# RANDOM FOREST
library(randomForest)
rf_bank_train <- randomForest(Attrition_Flag ~ .,
                           data = bank_train,
                           ntree = 500, 
                           mtry = 5,
                           importance = TRUE,
                         type = classification)

print(rf_bank_train)
plot(rf_bank_train)

rf_bank_test <- randomForest(Attrition_Flag ~ .,
                              data = bank_test,
                              ntree = 500, 
                              mtry = 5,
                              importance = TRUE,
                              type = classification)

print(rf_bank_test)
plot(rf_bank_test)

#ANALYZING ACCURACY: CM

#TRAIN -- Confusion Matrix:
TN = 5818
TP = 684
FP = 481
FN = 229

#accuracy = 
(TN + TP)/ (TN + TP + FP + FN)
#0.901 Accuracy
#--
#sensitivity = (TP) / (TP + FN)
(TP) / (TP + FN)
#Sensitivity = 0.749
#--
#specificity = (TN) / (TN + FP)
(TN) / (TN + FP)
#Specificity - 0.923
#--



#TEST -- Confusion Matrix:
TN = 1474
TP = 164
FP = 111
FN = 54

#accuracy = 
(TN + TP)/ (TN + TP + FP + FN)
#0.908 Accuracy
#--
#sensitivity = (TP) / (TP + FN)
(TP) / (TP + FN)
#Sensitivity = 0.752
#--
#specificity = (TN) / (TN + FP)
(TN) / (TN + FP)
#Specificity - 0.9299
#--

library(randomForestExplainer)
library(ggplot2)

# plot min depth distribution
plot_min_depth_distribution(rf_bank_train)




