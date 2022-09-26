Mitigation
==========

`holisticai.bias.mitigation` is a python module mitigating *bias* in algorithms. Our classes cover *pre-processing*, *post-processing* and *post-processing*.

.. _preprocessing:

Pre-processing
--------------
.. autosummary:: 
    :toctree: generated/

    holisticai.bias.mitigation.Reweighing
    holisticai.bias.mitigation.LearningFairRepresentation

.. _inprocessing:

In-processing
--------------
.. autosummary:: 
    :toctree: generated/

    holisticai.bias.mitigation.ExponentiatedGradientReduction
    holisticai.bias.mitigation.GridSearchReduction

.. _postprocessing:

Post-processing
---------------
.. autosummary:: 
    :toctree: generated/
    
    holisticai.bias.mitigation.CalibratedEqualizedOdds
    holisticai.bias.mitigation.EqualizedOdds
    holisticai.bias.mitigation.RejectOptionClassification
