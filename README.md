# cnmfe_judge
This is a convolutional neural network that classifies cnmfe outputs (cell-like components) 
into cell/not cell categories based on spatial footprint, pixel morphology and temporal trace characteristics. 

The skeleton architecture is based on the CNN used for 2p caiman processed data. This model has more regularization
steps compared to the component evaluator used for 2p data in caiman but is trained on 1p data. 
