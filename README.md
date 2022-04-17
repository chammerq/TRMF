
## Temporally Regularized Matrix Factorization
Functions to estimate temporally regularized matrix factorizations (TRMF) for forecasting and imputing values in short but high-dimensional time series. 
Uses regularized alternating least squares to compute the factorization, allows for several types of constraints on matrix factors and 
natively handles weighted and missing data. External regressors can also be included in the factorization. See documentation for more details.

## CRAN
This package is also available on CRAN: 

https://CRAN.R-project.org/package=TRMF


## How to use
To use the TRMF package to factor a time series matrix:

1. Create TRMF object for your time series matrix 
  ```{r,eval=FALSE}
obj = create_TRMF(A)
```
  a. It is recommended to scale the matrix using one of the scaling option in create_TRMF

  b. Missing values are imputed as default
  
2. Add a constraint and regularization for Fm to TRMF object
```{r,eval=FALSE}
obj = TRMF_columns(obj,reg_type = "nnls",lambda=1)
```
3. Add temporal regularization model for Xm to TRMF object
```{r,eval=FALSE}
obj = TRMF_trend(obj,numTS = 2,order = 2,lambdaD=1)
```
4. Maybe add another temporal regularization model for Xm to TRMF object
```{r,eval=FALSE}
obj = TRMF_trend(obj,numTS = 3,order = 0.5,lambdaD=10)
```

5. Maybe add an external regressor
```{r,eval=FALSE}
obj = TRMF_regression(obj, Xreg, type = "global")
```

6. Train object
```{r,eval=FALSE}
out = train(obj)
```

7. Evaluate solution
```{r,eval=FALSE}
summary(out)
plot(out)
resid(out)
fitted(out)
```

8. Get solution
```{r,eval=FALSE}
impute_TRMF(out)
coef(out)
Fm = out$Factors$Fm
Xm =out$Factors$Xm 
predict(out)
```

#### References:
The is package is loosely based on the following paper:

Yu, Hsiang-Fu, Nikhil Rao, and Inderjit S. Dhillon. "High-dimensional time series prediction with missing values." arXiv preprint arXiv:1509.08333 (2015).
