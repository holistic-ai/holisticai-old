from xml.dom import NotSupportedErr

from holisticai.utils.transformers.bias import (
    BIAS_TAGS,
    BMInprocessing,
    BMPostprocessing,
    BMPreprocessing,
)


class UTransformersHandler:
    def __init__(self, steps, params_hdl):
        """
        Initialize step groups and apply some validations.

        Description
        ----------
        Create bias mitigation groups for preprocessing, inprocessing and postprocessing strategies.
        Pipeline support only one postprocessing in the pipeline.

        Parameters
        ----------
        steps : list
            Pipeline steps

        params_hdl : UTransformersHandler
            Pipeline parameters handler during fit, fit_transform, transform function execution.
        """
        self.bias_mitigators_validation(steps)
        self.steps_groups = {
            tag: [step for step in steps if step[0].startswith(tag)]
            for tag in BIAS_TAGS
        }
        for tag, steps in self.steps_groups.items():
            for step in steps:
                step[1].link_parameters(params_hdl)

    def bias_mitigators_validation(self, steps):
        """Validate stem words and bias mitigator position in the pipeline"""
        tag2info = {
            BIAS_TAGS.PRE: BMPreprocessing,
            BIAS_TAGS.INP: BMInprocessing,
            BIAS_TAGS.POST: BMPostprocessing,
        }

        mitigator_groups_by_name = {
            tag: [step for step in steps if step[0].startswith(tag)]
            for tag in BIAS_TAGS
        }
        mitigator_groups_by_object = {
            tag: [step for step in steps if isinstance(step[1], tag2info[tag])]
            for tag in BIAS_TAGS
        }

        # Validate if all objects with bias mitigator stem (BIAS_TAGS) are linked with a correct Bias Mitigator Transformer objects
        if not mitigator_groups_by_name == mitigator_groups_by_object:
            raise NameError(
                f"Mitigator objects and name doesn't match, grouped by name: {mitigator_groups_by_name} \
                and grouped by object type:{mitigator_groups_by_object}"
            )

        num_post_mitigators = len(mitigator_groups_by_name[BIAS_TAGS.POST])

        # Validate if postprocessor bias mitigators are defined after classifier
        if num_post_mitigators > 0:
            post_classifier_step_names, _ = zip(
                *mitigator_groups_by_name[BIAS_TAGS.POST]
            )
            assert all(
                [name.startswith(BIAS_TAGS.POST) for name in post_classifier_step_names]
            ), f"Only bias mitigators postprocessor are supported, \
                utransformer postprocessors founded: {post_classifier_step_names}"

        # Validate that exists only one bias mitigator postprocessor
        # TODO: Evaluate in other cases.
        if not len(mitigator_groups_by_name[BIAS_TAGS.POST]) <= 1:
            raise NotSupportedErr("Pipeline supports max 1 postprocessor mitigator.")

    def drop_post_processing_steps(self, steps):
        """
        Drop post-processing steps from input steps list.

        Description
        ----------
        The function check the steps names and drop names with prefix 'bm_pos'.

        Parameters
        ----------
        steps : list
            Pipeline steps

        Returns
        -------
        list
            New steps list
        """
        return [step for step in steps if not step[0].startswith(BIAS_TAGS.POST)]

    @property
    def pos_processing_steps(self):
        """
        Return bias mitigation post-estimator steps.
        """
        return self.steps_groups[BIAS_TAGS.POST]

    def fit_postprocessing(self, Xt, y):
        """
        Fit the pos-estimator transformers.

        Description
        ----------
        Call `fit` of each post-estimator transformer in the pipeline.

        Parameters
        ----------
        Xt : numpy array
            Transformer input data

        y: numpy array
            Target vector

        Returns
        -------
        None
        """
        step = self.pos_processing_steps[0]
        step[1].fit(X=Xt, y_true=y)

    def transform_postprocessing(self, Xt):
        """
        Compute the transformed prediction vector with pos-estimator transformers.

        Description
        ----------
        Call `transform` of each post-estimator transformer in the pipeline.

        Parameters
        ----------
        Xt : list
            Transformer input data

        Returns
        -------
        dict
            Dictionaty with post-processed prediction vectors
        """
        step = self.pos_processing_steps[0]
        yt = step[1].transform(X=Xt)
        return yt


class ParametersHandler:
    def __init__(self, param_names=None, step_name=None):
        self.param_names = param_names
        self.step_name = step_name
        self.clean_parameters()

    def clean_parameters(self):
        self.dict_params = {}

    def set_shared_parameters(self, dict_params):
        self.dict_params = dict_params

    def __contains__(self, param_name):
        return param_name in self.dict_params

    def __getitem__(self, param_name):
        return self.dict_params[param_name]

    def __setitem__(self, param_name, param_value):
        self.dict_params[param_name] = param_value

    def feed(self, params, return_dropped=False):
        dropped_params = {}
        self.clean_parameters()
        if self.step_name:
            for name, value in params.items():
                if name.startswith(self.step_name):
                    param_name = name.split("__", 1)[1]
                    self[param_name] = value
                else:
                    dropped_params[name] = value

        elif self.param_names:

            for name, value in params.items():
                if name in self.param_names:
                    param_name = name.split("__", 1)[1]
                    self[param_name] = value
                else:
                    dropped_params[name] = value

        if return_dropped:
            return dropped_params


class PipelineParametersHandler:
    def __init__(self):
        self.bias_mitigator = ParametersHandler(
            param_names=["bm__group_a", "bm__group_b"]
        )

    def create_estimator_parameters(self, estimator_name, estimator):
        self.estimator = ParametersHandler(step_name=estimator_name)
        self.estimator_model = estimator

    def get_estimator_paramters(self):
        return self.estimator.dict_params
