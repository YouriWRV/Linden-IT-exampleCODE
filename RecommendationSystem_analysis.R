#You need to install two packages, one for the barplot and one to read excel files
#install.packages("ggplot2")
#install.packages("readxl")


#This should be the path to the excel file
dir_xlsxFile <- "uu_dingentjens/AI Master/methods in AI research/example_results.xlsx"

library(readxl)
data <- read_excel(dir_xlsxFile)
View(data)


firstSystem <- data[,2]
mean_satisfaction_per_participant_first_system <- rowMeans(data[,8:14])
mean_satisfaction_per_participant_second_system <- rowMeans(data[,15:21])

mean_secondSystem <- cbind(firstSystem,mean_satisfaction_per_participant_second_system)
mean_firstSystem <- cbind(firstSystem,mean_satisfaction_per_participant_first_system)
View(mean_firstSystem)
View(mean_secondSystem)

mean_firstSystemA <- mean_firstSystem[mean_firstSystem$firstSystem == "A",]
mean_firstSystemB <- mean_firstSystem[mean_firstSystem$firstSystem == "B",]

mean_secondSystemA <- mean_secondSystem[mean_secondSystem$firstSystem == "B",]
mean_secondSystemB <- mean_secondSystem[mean_secondSystem$firstSystem == "A",]


systemA <- c(unlist(mean_firstSystemA[,2]),  unlist(mean_secondSystemA[,2]))
systemB <- c(unlist(mean_firstSystemB[,2]),  unlist(mean_secondSystemB[,2]))
sd_systemA <- sd(systemA)
sd_systemB <- sd(systemB)

mean_perSystem <- data.frame(Satisfaction_score=c(mean(systemA), mean(systemB)), System=c("A","B"), sd=c(sd_systemA,sd_systemB))


library("ggplot2")
bp <- ggplot(data=mean_perSystem, aes(x=System, y=Satisfaction_score, fill=System)) +
      geom_bar(stat="identity")+
  geom_errorbar(aes(ymin=Satisfaction_score-sd, ymax=Satisfaction_score+sd), width=.2,
                position=position_dodge(.9))
     
bp+scale_fill_brewer(palette="Paired") + theme_minimal()
bp


wilcox.test(systemA,systemB)
