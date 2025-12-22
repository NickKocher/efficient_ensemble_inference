from auto_sklearn_search_space.pipeline.classification import SimpleClassificationPipeline
from auto_sklearn_search_space.pipeline.regression import SimpleRegressionPipeline

spaces = {
    "classification": SimpleClassificationPipeline,
    "regression": SimpleRegressionPipeline,
}
