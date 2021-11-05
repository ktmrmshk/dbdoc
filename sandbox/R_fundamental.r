# Databricks notebook source
c(-1, 3.1, 5)

# COMMAND ----------

as.matrix( c(-1, 3.1, 5) )

# COMMAND ----------

x = c(-1,3.1,5)

# COMMAND ----------

x <- c(-1, 3.1, 5)

# COMMAND ----------

x

# COMMAND ----------

x1 = 3+4i
x1

# COMMAND ----------

mode(x1)

# COMMAND ----------

x2=c("foobar", "R")
x2

# COMMAND ----------

mode(x2)

# COMMAND ----------

x3 = c(FALSE, TRUE, F, T, T, FALSE)
x3

# COMMAND ----------

mode(x3)

# COMMAND ----------

c(x1, x2)

# COMMAND ----------

f=c("a", "b", "c", "a", "c")
f <- factor(f)
f

# COMMAND ----------

mode(f)

# COMMAND ----------

class(f)

# COMMAND ----------

attributes(f)

# COMMAND ----------

mode( as.Date("2021-01-31", "%Y-%m-%d") )

# COMMAND ----------

a <- c(3, 5, NA, 12)
a[c(2, 3, 4)]

# COMMAND ----------

a[c(T,F,F,T)]

# COMMAND ----------

a>4

# COMMAND ----------

a1 = 1:4
a1

# COMMAND ----------

b1 = c(1,2,3,4)
b1

# COMMAND ----------

a2 <- 5:8

# COMMAND ----------

a1+a2

# COMMAND ----------

a1*a1

# COMMAND ----------

a1/a2

# COMMAND ----------

a=c(1:3, c(3,4,5), 3*(3:5))
a

# COMMAND ----------

A=matrix(a, nrow=3, ncol=3)
A

# COMMAND ----------

mode(A)

# COMMAND ----------

mode(a)

# COMMAND ----------

attributes(A)

# COMMAND ----------

class(A)

# COMMAND ----------

A[1,3]

# COMMAND ----------

A[1,]

# COMMAND ----------

A[,3]

# COMMAND ----------

diag(A)

# COMMAND ----------

A*A

# COMMAND ----------

A%*%A

# COMMAND ----------

data(iris)

# COMMAND ----------

iris

# COMMAND ----------

mode(iris)

# COMMAND ----------

class(iris)

# COMMAND ----------


