#pre-setup
setwd("/Users/kliu/Desktop/fb")
getwd()

#package
library(caret)
library(randomForest)
library(xgboost)
library(mlbench)
library(rpart)
library(e1071)

#read files
bids<-read.csv("bids.csv",header = TRUE, stringsAsFactors = FALSE)
test<-read.csv("test.csv",header = TRUE, stringsAsFactors = FALSE)
train<-read.csv("train.csv",header = TRUE, stringsAsFactors = FALSE)

#create features
bid_stats<-data.frame(matrix(ncol=20,nrow=0))
colnames(bid_stats)<-c("bidder_id", 
                        "num_bid",
                        "num_auction", 
                        "num_device",
                        "min_response_time", 
                        "mean_response_time",
                        "num_country", 
                        "num_ip",
                        "num_url",
                        "num_time",
                        "bid_per_device",
                        "bid_per_auction",
                        "bid_per_url",
                        "mean_ip",
                        "mean_url",
                        "max_country",
                        "mean_device",
                        "max_bid_per_url",
                        "mean_bid_per_ip",
                        "mean_bid_per_country")

bidder<-unique(bids$bidder_id)

for(i in (1:length(bidder))){
  input<-bids[bids$bidder_id == bidder[i], ]
  nbid<-aggregate(bid_id ~ bidder_id, data = input, length)
  nt<-aggregate(time ~ bidder_id, data = input, function(x) length(unique(x)))
  na<-aggregate(auction ~ bidder_id,data =input, function(x) length(unique(x)))
  nd<-aggregate(device ~ bidder_id,data =input, function(x) length(unique(x)))
  nc<-aggregate(country ~ bidder_id,data =input, function(x) length(unique(x)))
  nip<-aggregate(ip ~ bidder_id,data =input, function(x) length(unique(x)))
  nurl<-aggregate(url ~ bidder_id,data =input, function(x) length(unique(x)))
  bid_pdev<-aggregate(bid_id ~ device, data = input, function(x) length(unique(x)))
  bid_pauc<-aggregate(bid_id ~ auction, data = input, function(x) length(unique(x)))
  bid_purl<-aggregate(bid_id ~ url, data = input, function(x) length(unique(x)))
  bid_pip<-aggregate(bid_id ~ ip, data = input, function(x) length(unique(x)))
  bid_pcountry<-aggregate(bid_id ~ country, data = input, function(x) length(unique(x)))
  mean_ip<-aggregate(bid_id ~ip,data=input,FUN=length)
  mean_url<-aggregate(bid_id ~url,data=input,FUN=length)
  mean_country<-aggregate(bid_id ~country,data=input,FUN=length)
  mean_device<-aggregate(bid_id ~ device,data = input,FUN = length)
  if (length(input$time)>1){
    min_time<-min(diff(input$time))
    mean_time<-mean(diff(input$time))
  }else{
    min_time <-NA
    mean_time <- NA
  }
  bid_stats[i,1]<-input$bidder_id[1]
  bid_stats[i,2]<-nbid$bid_id
  bid_stats[i,3]<-na$auction
  bid_stats[i,4]<-nd$device
  bid_stats[i,5]<-min_time
  bid_stats[i,6]<-mean_time
  bid_stats[i,7]<-nc$country
  bid_stats[i,8]<-nip$ip
  bid_stats[i,9]<-nurl$url
  bid_stats[i,10]<-nt$time
  bid_stats[i,11]<-mean(bid_pdev$bid_id)
  bid_stats[i,12]<-mean(bid_pauc$bid_id)
  bid_stats[i,13]<-mean(bid_purl$bid_id)
  bid_stats[i,14]<-mean(mean_ip$bid_id)
  bid_stats[i,15]<-mean(mean_url$bid_id)
  bid_stats[i,16]<-max(mean_country$bid_id)
  bid_stats[i,17]<-mean(mean_device$bid_id)
  bid_stats[i,18]<-max(bid_purl$bid_id)
  bid_stats[i,19]<-mean(bid_pip$bid_id)
  bid_stats[i,20]<-mean(bid_pcountry$bid_id)
}

mypca<-prcomp(bid_stat[5:17],center = TRUE,scale. = TRUE)

summary(mypca)

plot(mypca, type = "l")

bid_stat<-merge(train,bid_stats, by="bidder_id", all.x = TRUE)

human_mean_time<-mean(bid_stat[bid_stat$outcome == 0,"mean_response_time"],na.rm = TRUE)

bot_mean_time<-mean(bid_stat[bid_stat$outcome == 1,"mean_response_time"],na.rm = TRUE)

human_min_time<-mean(bid_stat[bid_stat$outcome == 0,"min_response_time"],na.rm = TRUE)

bot_min_time<-mean(bid_stat[bid_stat$outcome == 1,"min_response_time"],na.rm = TRUE)

for(i in (1:nrow(bid_stat))){
  if (is.na(bid_stat$min_response_time[i]) & bid_stat$outcome[i] == 0){
    bid_stat$min_response_time[i]<-human_min_time
  }
  if (is.na(bid_stat$min_response_time[i]) & bid_stat$outcome[i] == 1){
    bid_stat$min_response_time[i]<-bot_min_time
  }
  if (is.na(bid_stat$mean_response_time[i]) & bid_stat$outcome[i] == 0){
    bid_stat$mean_response_time[i]<-human_mean_time
  }
  if (is.na(bid_stat$mean_response_time[i]) & bid_stat$outcome[i] == 1){
    bid_stat$mean_response_time[i]<-bot_mean_time
  }
}

bid_stat<-bid_stat[!is.na(bid_stat$num_bid),]

bid_stat$outcome<-as.factor(bid_stat$outcome)

levels(bid_stat$outcome)<-c("human","bot")

#smp_size <- floor(0.7 * nrow(bid_stat))

#set.seed(10011111)

#train_ind <- sample(seq_len(nrow(bid_stat)), size = smp_size)

#bid_stat_train <- bid_stat[train_ind, ]
#bid_stat_test <- bid_stat[-train_ind, ]

bid_stat_train<-bid_stat

mycontrol<-trainControl(method = "repeatedcv", number = 50,savePredictions = 'final',classProbs = TRUE)

model<-train(outcome ~ num_bid
                      +num_auction
                      +num_device
                      +min_response_time
                      +mean_response_time
                      +num_country
                      +num_ip
                      +num_url
                      +num_time
                      +bid_per_device
                      +bid_per_auction
                      +bid_per_url
                      +mean_ip
                      +mean_url
                      +max_country
                      +mean_device
                      +max_bid_per_url
                      +mean_bid_per_ip
                      +mean_bid_per_country
                      ,data = bid_stat_train, method = "rf", trControl=mycontrol,tuneLength = 3)

print(varImp(model,scale=FALSE))

model$finalModel

#########################################################
model2<-train(outcome ~ num_bid
             +num_auction
             +num_device
             +min_response_time
             +mean_response_time
             +num_country
             +num_ip
             +num_url
             +num_time
             +bid_per_device
             +bid_per_auction
             +bid_per_url
             +mean_ip
             +mean_url
             +max_country
             +mean_device
             +max_bid_per_url
             +mean_bid_per_ip
             +mean_bid_per_country
             ,data = bid_stat_train, method = "AdaBag", trControl=mycontrol)

print(varImp(model2,scale=FALSE))

model2$finalModel

str(bid_stat)

######################################
model3<-train(outcome ~ num_bid
              +num_auction
              +num_device
              +min_response_time
              +mean_response_time
              +num_country
              +num_ip
              +num_url
              +num_time
              +bid_per_device
              +bid_per_auction
              +bid_per_url
              +mean_ip
              +mean_url
              +max_country
              +mean_device
              +max_bid_per_url
              +mean_bid_per_ip
              +mean_bid_per_country
              ,data = bid_stat_train, method = "ada", trControl=mycontrol)

print(varImp(model3,scale=FALSE))

model3$finalModel

######################################

model2<-train(outcome ~ num_bid
              +num_auction
              +num_device
              +min_response_time
              +mean_response_time
              +num_country
              +num_ip
              +num_url
              +num_time
              +bid_per_device
              +bid_per_auction
              +bid_per_url
              +mean_ip
              +mean_url
              +max_country
              +mean_device
              +max_bid_per_url
              +mean_bid_per_ip
              +mean_bid_per_country
              ,data = bid_stat_train, method = "xgbTree", trControl=mycontrol)

print(varImp(model4,scale=FALSE))

model4$finalModel
###############################################

######################################

model5<-train(outcome ~ num_bid
              +num_auction
              +num_device
              +min_response_time
              +mean_response_time
              +num_country
              +num_ip
              +num_url
              +num_time
              +bid_per_device
              +bid_per_auction
              +bid_per_url
              #+mean_ip
              #+mean_url
              +max_country
              +mean_device
              +max_bid_per_url
              +mean_bid_per_ip
              +mean_bid_per_country
              ,data = bid_stat_train, method = "randomGLM", trControl=mycontrol)

print(varImp(model5,scale=FALSE))

model5$finalModel
###############################################

#get weak learner results
#bid_stat_train$model5_results<-model5$pred$bot[order(model5$pred$rowIndex)]
bid_stat_train$model4_results<-model4$pred$bot[order(model4$pred$rowIndex)]
bid_stat_train$model3_results<-model3$pred$bot[order(model3$pred$rowIndex)]
bid_stat_train$model2_results<-model2$pred$bot[order(model2$pred$rowIndex)]
bid_stat_train$model1_results<-model$pred$bot[order(model$pred$rowIndex)]


#create the predictor string
predictors<-c("num_bid",
              "num_auction",
              "num_device",
              "min_response_time",
              "mean_response_time",
              "num_country",
              "num_ip",
              "num_url",
              "num_time",
              "bid_per_device",
              "bid_per_auction",
              "bid_per_url",
              "mean_ip",
              "mean_url",
              "max_country",
              "mean_device",
              "max_bid_per_url",
              "mean_bid_per_ip",
              "mean_bid_per_country")

#create the second level predictors based on the results of the weak learner
predictor_top<-c("model1_results","model2_results","model3_results")

#build the second level model
top_model2<-train(bid_stat_train[,predictor_top],bid_stat_train$outcome,method = "ada", trControl=mycontrol)

top_model<-train(bid_stat_train[,predictor_top],bid_stat_train$outcome,method = "rf", trControl=mycontrol)

top_model2$finalModel

top_model$finalModel
#########################################################
bid_stat_pred<-merge(test,bid_stats,by="bidder_id",all.x = TRUE)



for(i in (1:nrow(bid_stat_pred))){
  if (is.na(bid_stat_pred$min_response_time[i]) & is.na(bid_stat_pred$mean_response_time[i])){
    bid_stat_pred$min_response_time[i]<-human_min_time
    bid_stat_pred$mean_response_time[i]<-human_mean_time
  }
}

no_record_bidder<-cbind("bidder_id" = bid_stat_pred[is.na(bid_stat_pred$num_bid),][,1],"prediction" = rep(0,nrow(bid_stat_pred[is.na(bid_stat_pred$num_bid),])))

nrow(no_record_bidder)

bid_stat_pred<-bid_stat_pred[!is.na(bid_stat_pred$num_bid),]

length(predict(model,bid_stat_pred[,predictors], type = "prob")$bot)

bid_stat_pred$model1_results<-predict(model,bid_stat_pred[,predictors], type = "prob")$bot

bid_stat_pred$model2_results<-predict(model2,bid_stat_pred[,predictors], type = "prob")$bot

bid_stat_pred$model3_results<-predict(model3,bid_stat_pred[,predictors], type = "prob")$bot

bid_stat_pred$model4_results<-predict(model4,bid_stat_pred[,predictors], type = "prob")$bot

#bid_stat_pred$model5_results<-predict(model5,bid_stat_pred[,predictors], type = "prob")$bot

d<-predict(top_model2, bid_stat_pred[,predictor_top],type = "prob")

e<-predict(top_model, bid_stat_pred[,predictor_top],type = "prob")


D<-data.frame(matrix(ncol=2,nrow=0))
for(i in (1:nrow(d))){
  D[i,1]<-mean(d[i,1],e[i,1])
  D[i,2]<-mean(d[i,2],e[i,2])
}

###############################################

pred<-cbind("bidder_id" = bid_stat_pred[,1],"prediction" = D[,2])

results<-as.data.frame(rbind(pred,no_record_bidder),stringsAsFactors = FALSE)

nrow(no_record_bidder)

#results<-as.data.frame(pred, stringsAsFactors = FALSE)

results$prediction<-as.numeric(results$prediction)

test_order=as.data.frame(cbind("bidder_id"=test$bidder_id,"index" = row.names(test)),stringsAsFactors = FALSE)

test_order$index<-as.numeric(test_order$index)

results<-merge(results, test_order,by = "bidder_id",all.x = TRUE)

results<-results[order(results$index),][,c(1,2)]

nrow(results)

write.csv(results, "submit.csv", row.names = FALSE)

###########################

