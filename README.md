# Scaffolding: Syntactic Scaffolds for Semantic Structures

A repository based on the EMNLP 2018 [paper](https://arxiv.org/abs/1808.10485) for Frame-Semantic and PropBank Semantic Role Labeling with Syntactic Scaffolding. Code for coreference with a syntactic scaffold coming soon.

## Installation
This repository was built on an earlier version of [AllenNLP](https://github.com/allenai/allennlp).
Due to changes in the API, we recommended installing directly via steps below (adapted from the AllenNLP installation), as opposed to using an installed version of AllenNLP.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```
    conda create -n scaffold python=3.6
    ```

3.  Activate the Conda environment.  (You will need to activate the Conda environment in each terminal in which you want to use AllenNLP.

    ```
    source activate scaffold
    ```

4. Install in your environment

    ```
    git clone https://github.com/swabhs/scaffolding.git
    ```

5. Change your directory to where you cloned the files:

    ```
    cd scaffolding/
    ```

6.  Install the required dependencies.

    ```
    INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh
    ```

7. Install PyTorch version 0.3 **via pip** ([modify](https://pytorch.org/previous-versions/) based on your CUDA environment).
Conda-based installation results in [slower rutime](https://github.com/pytorch/pytorch/issues/537) because of a CUDA issue.

    ```
    pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
    ```

## Step 1: Get Data

Download [FN data](https://drive.google.com/file/d/15n3M4AmURGdGqnNAjn352buUTV5S-fVI/view?usp=sharing) and place it under a `data/` directory under the root directory.

## Step 2: Get Pre-trained Models

[Frame-SRL baseline](https://drive.google.com/open?id=1f7ZLOBc65Y74hPQlYY8mGVHkCiN14dwH)

[Frame-SRL scaffold with common non-terminals](https://drive.google.com/open?id=1V1-U70U-wDKaG3zuONQN3eB77jjS3FSd)

[PropBank SRL baseline](https://drive.google.com/file/d/1ThTagaJeZkIACEYfDn6f4mjMHs0PMSAo/view?usp=sharing)

[PropBank SRL scaffold with common non-terminals](https://drive.google.com/file/d/1vv3KC_OLx0A7ItKqBz9yWwZLIsVZhR5B/view?usp=sharing)


## Step 3: Test

```
python -m allennlp.run evaluate \
    --archive-file log_final_pb_baseline/model.tar.gz \
    --evaluation-data-file data/fndata-1.5/test/fulltext/ \
    --cuda-device 0
```

Coming Soon: 
For the syntactic scaffold model for PropBank SRL, use the `pbscaf` branch:
```
git checkout pbscaf
```
and then run evaluation as above.

## Training

For scaffolds, use `$command=train_m` and for baselines, `$command=train`.
```
python -m allennlp.run $command training_config/$config --serialization-dir log
```

## Acknowledgment

Paper coming soon on ACL Anthology / ArXiv.

```
@inproceedings{swayamdipta:2018,
    author      = {Swabha Swayamdipta and Sam Thomson and Kenton Lee and Luke Zettlemoyer and Chris Dyer and Noah A. Smith},
    title       = {Syntactic Scaffolding for Semantic Structures},
    booktitle   = {Proc. of EMNLP},
    url         = {https://arxiv.org/abs/1808.10485},
    year        = {2018}
}
```

