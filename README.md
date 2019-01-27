# Benchmarking Pool-Based Active Learning
Check out these [plots comparing active learning algorithms](results/outline.html)


#Results



20-NewsGroups
-------------
![20newsgroups](results/plots/20newsgroups_vectorized-logistic_regression.png)

Boston
-------------
![boston](results/plots/boston-logistic_regression.png)

California Housing
------------------
![california_housing](results/plots/california_housing-logistic_regression.png)




# Install
- Clone the repo

(We assume you're on python3 anaconda)


# Running
To run the active learning evaluation, type:

    ./run.sh

# Pushing a new algorithm



# What is this
Goal of puddle is to provide:
* an interface for implementing model-agnostic pool-based example-selector algorithms, for pool-based classification tasks
* a set of benchmarks to evaluate these selectors.  Benchmarks include specific datasets (vision, categorical, NLP), as well as specific models that perform well on these datasets

# Todo
* constrain goal of puddle formally ^.  What do we assume feature space representation is for modeling?  what do selectors have access to?  
    => maybe ability to get nearest neighbors?
* get this to run again (or find version that ran)
* add bert lstm classification benchmark
* add semi-supervised as well as active learning.  end-goal is pool-based transfer learning
* more benchmarks
    * amazon reviews?  
    * NER
    * image segmentation over satellite imagery (unet, spacenet etc?)?
    * standard image classification
    => goal is to be able to run these benchmarks efficiently

# References
* https://arxiv.org/pdf/1804.09170.pdf
* https://github.com/erikbern/ann-benchmarks
* burr settles active learning summary: http://burrsettles.com/pub/settles.activelearning.pdf
* some active learning py implementations: https://github.com/davefernig/alp

