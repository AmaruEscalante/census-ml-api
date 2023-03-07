# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Developed by Amaru Escalante, 2023, v1.
- Convolutional Neural Net.
## Intended Use
- The model is intended to be used to classify a persons income as either above or below 50k a year by using a variety of features such as age, education, and occupation.
- Intended use of this model is to showcase the use of MLOps Tools such as DVC in a production environment.
## Training Data
- The training data was obtained from the UCI Machine Learning Repository.

## Evaluation Data
- The evaluation data was obtained from the UCI Machine Learning Repository.
## Metrics
<!-- _Please include the metrics used and your model's performance on those metrics._ -->
- Precision: 0.7348314606741573
- Recall: 0.6244430299172502
- Fbeta b=1: 0.6751548520302821

## Ethical Considerations
It is important to consider the potential ethical implications of using this model in practice. The model's predictions may impact a person's job prospects, financial stability, and other important life outcomes. As such, it is important to ensure that the model is fair and unbiased, and that it does not unfairly discriminate against any particular groups or individuals. 

## Caveats and Recommendations
- The model may not be suitable for use in many populations since only a handful of countries were represented in the dataset.
- The model is limited by the diversity of the dataset in terms of peoples backgrounds which some have more representation than others. 
- The dataset was collected in the 1990s and may not be representative of the current population.