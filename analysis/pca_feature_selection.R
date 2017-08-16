setwd('~/repositories/ada/data/')
data <- read.csv('numerai_training_data.csv')
feature_matrix<-data[,4:25]
values<-feature_matrix[,1:(ncol(feature_matrix)-1)]
feature_matrix_pca<-prcomp(values,center = TRUE,scale. = TRUE)

summary(feature_matrix_pca)
target_cor<-apply(feature_matrix_pca$x, 2, 
                  function(x){p.val<-cor.test(feature_matrix$target, x)$p.value})
target_ttest<-apply(feature_matrix_pca$x, 2, 
                    function(x){p.val<-t.test(x~feature_matrix$target)$p.value})
p.adjust(target_ttest, method="BH")
# PC1          PC2          PC3          PC4          PC5          PC6          PC7          PC8 
# 1.305949e-03 4.868415e-01 7.880576e-07 2.966921e-05 1.284648e-10 1.284648e-10 4.116605e-03 4.868415e-01 
# PC9         PC10         PC11         PC12         PC13         PC14         PC15         PC16 
# 4.868415e-01 1.575630e-01 5.457941e-01 5.645201e-01 4.868415e-01 6.053423e-02 1.126122e-01 6.389793e-01 
# PC17         PC18         PC19         PC20         PC21 
# 4.868415e-01 8.527394e-01 4.868415e-01 4.868415e-01 9.243361e-01 

library(devtools)
library(ggplot2)
library(ggbiplot)
pc_biplot<-function(pca_object, pc1, pc2){
  g <- ggbiplot(pca_object, choices=pc1:pc2, obs.scale = 1, var.scale = 1, 
                groups = labels, ellipse = TRUE, 
                circle = TRUE)
  g <- g + scale_color_discrete(name = '')
  g <- g + theme(legend.direction = 'horizontal', 
                 legend.position = 'top')
  return(g)
}



plot_rotations<-function(prcomp_object, labels, pc){
  cor_data<-as.data.frame(prcomp_object$x[,pc])
  colnames(cor_data)<-"rotations"
  cor_data$labels<-labels
  cor_data$dummy_val<-0
  
  library(ggplot2)
  p_scatter<-ggplot(cor_data, aes(x=rotations, y=dummy_val, color=labels))+geom_point()+
    scale_y_continuous(limits=c(-0.05, 0.05))
  p_density<-ggplot(cor_data, aes(x=rotations, fill=labels))+
    geom_density(alpha=.3)
  print(p_density)
}

plot_x_dist<-function(prcomp_object, pc, target){
  pc_x = prcomp_object$x[,pc]
  pc_df = as.data.frame(cbind(pc_x, feature_matrix$target))
  colnames(pc_df) = c("pc_x", "target")
  pc_df$target[pc_df$target==0]='A'
  pc_df$target[pc_df$target==1]='B'
  p=ggplot(pc_df, aes(x=pc_x, fill=target)) + geom_density(alpha=.3)
  return(p)
}

plot_x_dist(feature_matrix_pca, 7, feature_matrix$target)

p<-plot_x_dist(feature_matrix_pca, 1, feature_matrix$target)

plot_rotations(feature_matrix_pca, feature_matrix$target, 1)
pc_biplot(feature_matrix_pca, 1,2)

ggbiplot(feature_matrix_pca, choices=1:2, obs.scale = 1, var.scale = 1, 
         groups = labels, ellipse = TRUE, 
         circle = TRUE)

ggbiplot(pca_object, choices=pc1:pc2, obs.scale = 1, var.scale = 1, 
         groups = labels, ellipse = TRUE, 
         circle = TRUE)

head(feature.matrix.pca$rotation)
pc1.rotation <- feature.matrix.pca$rotation[,1]
holdout.values <- holdout[,1:ncol(holdout)-1]
holdout.values.norm<-scale(holdout.values, center=TRUE, scale=TRUE)
holdout.rotated.pc1 <- as.matrix(holdout.values.norm) %*% as.vector(pc1.rotation)
holdout.rotated <- data.frame(value=holdout.rotated.pc1, labels = holdout[,ncol(holdout)])
head(holdout.rotated)
#colnames(holdout.rotated)<-c('value', 'labels')
ggplot(holdout.rotated, aes(x=value, fill=labels))+geom_density(alpha=.3)


values.norm<-scale(values, center=TRUE, scale=TRUE)
training.rotated.pc1 <- as.matrix(values.norm) %*% as.vector(pc1.rotation)
head(training.rotated.pc1)
head(feature.matrix.pca$x[,1])


# do PCA for the trainig and holdout sets together
# combine the data from training and holdout
feature_matrix$ind<-'train'
holdout$ind<-'holdout'
all_data<-rbind(feature_matrix, holdout)
# write a version of non-centered data
write.csv(all_data, "all_data_raw.csv")
# create a centered version of the data and save it
all_values<-all_data[, 1:(ncol(all_data)-2)]
all_data_norm<-scale(all_values, center=TRUE, scale=TRUE)
all_data_norm<-as.data.frame(all_data_norm)
all_data_norm$labels<-all_data$labels
all_data_norm$ind<-all_data$ind
write.csv(all_data_norm, "all_data_norm.csv")


# load in the data with flight sequence and tail information attached
# do PCA and see if there is a separaetion by tail
setwd('C:/Users/ruiwei.jiang/workspace/repository/intermediate/stl_feature_matrix_train/')
all_data<-read.csv("all_data_selected.csv")
rownames(all_data)<-all_data$flight_seq
all_data<-all_data[, !colnames(all_data) == 'flight_seq']
# add in an identifier for airlines
for(i in 1:nrow(all_data)){
  all_data$airline[i]<-substr(as.character(all_data$tail[i]), 2,2)
}
all_values<-all_data[,!colnames(all_data)%in%c("labels", "tail", "flight_seq.1", "airline")]
all.pca<-prcomp(all_values,center = TRUE,scale. = TRUE)
tails<-all_data$tail
labels<-all_data$labels
airline<-all_data$airline

summary(all.pca)
plot_rotations(all.pca, airline, 1)
plot_rotations(all.pca, labels, 1)
pc_biplot(all.pca, 1,2)


# statistical testing between the means of the TRUE and FALSE groups
features<-colnames(all_values)
pvalues = as.data.frame(features)
pvalues$pval = NA
for(feature in features){
  test_data<-as.data.frame(all_values[,feature])
  test_data$labels<-all_data$labels
  colnames(test_data)<-c('value', 'labels')
  ttest = t.test(test_data$value[test_data$labels == 'False'], test_data$value[test_data$labels == 'True'])
  pvalues$pval[pvalues$features==feature]<-ttest$p.value
}

# statistical testing between means of TRUE and FALSE groups with sampling
features<-colnames(all_values)
sampling_pvalues <- as.data.frame(features)
sampling_pvalues$pval<-NA
n = 500
for(feature in features){
  test_data<-as.data.frame(all_values[,feature])
  test_data$labels<-all_data$labels
  colnames(test_data)<-c('value', 'labels')
  sig_count = 0
  for(i in 1:n){
    false_sample <- test_data[ sample( which( test_data$labels != "True" ) , 
                                       sum(test_data$labels=='True') ) , ]
    sample_data <- rbind(false_sample, test_data[test_data$labels=='True',])
    ttest = t.test(sample_data$value[sample_data$labels == 'False'], sample_data$value[sample_data$labels == 'True'])
    if (ttest$p.value<0.05){
      sig_count = sig_count + 1
    }
  }
  sampling_pvalues$pval[sampling_pvalues$features==feature]<-(n-sig_count)/(n-1)
  print(feature)
}
sampling_pvalues$adj.pval<-p.adjust(sampling_pvalues$pval, method='BH')


# calculate euclidean distance of each negative vs. negative, and then positive vs. each negative 
# entire matrix was too big to calculate; reuse the resampling technique to approximate
# the distance among negatives, and between negative and positives
library(pdist)
n = 500
true_obs <- rownames(all_data[which(all_data$labels == "True"),])
values_true<-all_data[rownames(all_data)%in%true_obs,!colnames(sample_data)%in%
                        c("labels", "tail", "flight_seq.1", "airline")]

false_obs<-rownames(all_data[which(all_data$labels == "False"),])
false_obs_samples<-list()
for (i in 1:100){
  if(length(false_obs)<705){
    data_sample<-all_data[rownames(all_data)%in%false_obs,!colnames(all_data)%in%
                            c("labels", "tail", "flight_seq.1", "airline")]
    false_obs_samples[[i]]<-data_sample
  }else{
    obs_sampling<-sample(false_obs, 705, replace=FALSE)
    data_sample<-all_data[rownames(all_data)%in%obs_sampling,!colnames(all_data)%in%
                            c("labels", "tail", "flight_seq.1", "airline")]
    false_obs_samples[[i]]<-data_sample
    false_obs<-false_obs[!false_obs%in%obs_sampling]
  }
  print(length(false_obs))
  print(dim(data_sample))
}

dist_f_f <- c()
dist_f_t<-c()
for(i in 1:length(false_obs_samples)){
  # sample the same number of negative cases as there are positive cases
  values_false_i<-false_obs_samples[[i]]
  sample_dist_f_t <- as.vector(dist(pdist(values_false_i, values_true)))
  dist_f_t<-c(dist_f_t, sample_dist_f_t)
  
  if(i<90){
    sampling <- sample(i:length(false_obs_samples), 10)
  }else{
    sampling<-i:length(false_obs_samples)
  }
  for(j in sampling){
    print(j)
    if(j==i){
      sample_dist_f_f<-dist(values_false_i, method="euclidean")
      dist_f_f<-c(dist_f_f, sample_dist_f_f)
    }else{
      values_false_j<-false_obs_samples[[j]]
      sample_dist_f_f<-as.vector(dist(pdist(values_false_i, values_false_j)))
      dist_f_f<-c(dist_f_f, sample_dist_f_f)
    }
  }
  print('another')
}
all_dist<-as.data.frame(c(dist_f_f, dist_f_t))

colnames(all_dist)<-'values'
all_dist$ind<-c(rep('false', length(dist_f_f)), rep('true', length(dist_f_t)))
ggplot(all_dist, aes(x=values, fill=ind)) +
  geom_histogram(binwidth=5, alpha=.5, position="identity")
ggplot(all_dist, aes(x=values)) +
  geom_histogram(binwidth=5, colour="black", fill="white")+
  facet_grid(ind ~ .)
# ks test to see the difference between the true and false distributions
ks.test(dist_f_f, dist_f_t)
# Two-sample Kolmogorov-Smirnov test
# 
# data:  dist_f_f and dist_f_t
# D = 0.73948, p-value < 2.2e-16
# alternative hypothesis: two-sided

sub_dist<-all_dist[all_dist$values<22,]
ggplot(sub_dist, aes(x=values)) +
  geom_histogram(binwidth=1, colour="black", fill="white")+
  facet_grid(ind ~ .)


subset<-all_data[, colnames(all_data)%in%c("X.u.SRVCE_SN01...u.Tire10_Press...u.ccca..", "labels")]
colnames(subset)<-c('value', 'labels')
ggplot(subset, aes(x=labels, y=value)) +
  geom_boxplot()