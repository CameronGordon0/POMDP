# POMDP

This project extends a python POMDPX parser (Slosic and Petric (2017) https://github.com/larics/python-pomdp) to handle special characters (*, -) and special terms ('identity', 'uniform') consistent with the POMDPX File Format as documented at https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation. A POMDP simulator may be called through the 'simulator_main_class_refactor.py' module. To run the POMDPX parser by itself call parser_main.py. Running testing_module will run automated testing of multiple network architectures. Parameters should be modified directly in the respective files. 

Code is included for the following algorithms: 
* Action-specific Deep Recurrent Q-Networks ADQN / ADRQN (Zhu et al. 2017) https://arxiv.org/pdf/1704.07978.pdf 
* Deep Recurrent Q-Networks DRQN (Hausknecht and Stone 2015) https://arxiv.org/abs/1507.06527 
* Deep Q-Networks (Mnih et al. 2013) https://arxiv.org/pdf/1312.5602.pdf 
* An extension to ADQN / ADRQN that includes the step reward in the observation-action history, that has been termed RADQN / RADRQN. 

Code options are included for the following technical modifications: 
* Prioritised Experience Replay (Schaul et al. 2015) https://arxiv.org/abs/1511.05952 
* Expert Buffers (various, examples of demonstration learning are given in (Hester et al. 2017) https://arxiv.org/abs/1704.03732
* Flooding regularization (Ishida et al. 2020) https://arxiv.org/abs/2002.08709 

A converter is included for POMDPX to OpenAI Gym environments (described in Brockman et al. 2016 https://arxiv.org/abs/1606.01540). Full integration with OpenAI Baselines https://github.com/openai/baselines has not been achieved, and remains a natural extension to the project. 

Modules for automated production of results, diagnostics, and visualisations are included. Data from recorded tests is included in the 'Results' folder. 

Modules are included for the construction of expert buffers from outputs from the DESPOT POMDP solver described in Ye et al. (2017) code: (https://github.com/AdaCompNUS/despot). These may be passed to initialise the memory buffer for the POMDP simulator. 

Modules for the creating of OpenAI Gym environments from POMDPX files are contained in the custom_gym folder. Integration with OpenAI Baselines has not yet been achieved.

Developed by Cameron Gordon at the University of Queensland, supervised by Dr. Nan Ye as part of a thesis for a Masters of Science (Mathematics).
