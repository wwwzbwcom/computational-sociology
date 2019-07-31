library(readstata13)

dta_name = "game1_test_data"

print(paste(dta_name, ".dta", sep=""))
df = read.dta13(paste(dta_name, ".dta", sep=""))

con<-file(paste(dta_name, ".csv", sep=""), encoding="UTF-8")

write.csv(df, file=con)
df = data.matrix(df)

write.csv(df, file=paste(dta_name, "_numeric.csv", sep=""))