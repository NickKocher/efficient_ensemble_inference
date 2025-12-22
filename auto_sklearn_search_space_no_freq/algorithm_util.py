from auto_sklearn_search_space_no_freq.pipeline.classification import SimpleClassificationPipeline
from auto_sklearn_search_space_no_freq.pipeline.regression import SimpleRegressionPipeline

spaces = {
    "classification": SimpleClassificationPipeline,
    "regression": SimpleRegressionPipeline,
}
