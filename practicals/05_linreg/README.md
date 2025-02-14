# Linear regression practical

In the practical session we discussed the [practical.ipynb](practical.ipynb) notebook.

## Regularization and standarized linear regression models

The predictions of the linear regression model to data not used to estimate its coefficients were poor. The normalized mean-square error (NMSE) was larger than 1.0, indicating that using the mean to predict all values is better than using the model predictions. Overfitting could be a cause of the poor predictions. One approach to address overfitting is regularization. We used regularization in the notebook [practical_regularised.ipynb](practical_regularised.ipynb). We had to use a regularization parameter larger than 1e10 to obtain NMSE's smaller than one. We had to use so large regularization parameters because the effect of the regularization parameter depends on the scale of the independent variables.
The scale of the ECoG data is large (min value=-24,712, max value=22,496), and this large scale requires use very large regularization parameter values.

To make the effect of the regularization parameter independent of the data scale, we standardized the predictors (i.e., removed the sample mean of each channel and divided by their sample standard deviation). We did so in the notebook
[practical_regularised_scaled.ipynb](practical_regularised_scaled.ipynb), where we obtained a NMSE of 0.77 with a regularization parameter of 1e4. This notebooks also states, but does not prove (exercise), the equivalence between standarized and non-standarized linear regression models.

These examples demonstrate a useful application of standardized linear regression models, and illustrate a beneficial effect of regularizationo.
