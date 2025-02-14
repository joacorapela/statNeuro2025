# Linear regression practical

In the practical session we discussed the [practical.ipynb](practical.ipynb) notebook.

The predicted ECoG activity appeared noisy. One approach to reduce this noise is to regularise the estimated coefficients. We did so in the [practical_regularised.ipynb](practical_regularised.ipynb) notebook, with regularisation parameters as large as 1e6. However, despite this large regularization, the prediction did not change much.

The problem is that the effect of the regularisation parameter depends on the scale of the predictors. The scale of the ECoG data is large (min value=-24,712, max value=22,496), and for this scale a regularization parameter of 1e6 is weak. Therefore, to make the effect of the regularization parameter independent of the data scale, we standardized the predictors (i.e., removed the sample mean of each channel and divided by the sample standard deviation of each channel). We did so in the notebook
[practical_regularised_scaled.ipynb](practical_regularised_scaled.ipynb), that shows that a regularization parameter of 1e5 reduces the predictions noise, but also augments their bias. This notebooks also states, but does not prove, the equivalence between standarized and non-standarized linear regression models.

These examples demonstrate a useful application of standardized linear regression models, and illustrate effect of strong regularisation on predictions of linear models.
