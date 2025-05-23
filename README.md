# Statistical Neuroscience 2025 repository

- lectures and practicals will take place at the Sainsbury Wellcome Centre lecture theatre. Lectures will be on Mondays from 1pm to 3pm and practicals on Fridays from 2pm to 3:30pm.

- material related to lectures can be found [here](https://github.com/joacorapela/statsNeuro2025/tree/master/lectures)

- material related to discussion sessions can be found [here](https://github.com/joacorapela/statsNeuro2025/tree/master/practicals)

- worksheets can be found [here](https://github.com/joacorapela/statsNeuro2025/tree/master/worksheets)

- <a name="lecturesSchedule"></a>lectures schedule:

    | Week | Date  | Topic | Lecturers | Type |
    |------|-------|-------|-----------|------|
    | 01 | Jan 13 | [Probability for temporal time series analysis](https://github.com/joacorapela/statNeuro2025/blob/master/lectures/01_temporalTimeSeriesAnalysis/temporalTimeSeriesAnalysis.pdf) | [Joaquin Rapela](https://www.gatsby.ucl.ac.uk/~rapela) | lecture |
    | 01 | Jan 17 | [Probability for temporal time series analysis](https://github.com/joacorapela/statNeuro2025/blob/master/practicals/01_temporalTimeSeriesAnalysis/practical_temporalTimeSeriesAnalysis.pdf) | [Joaquin Rapela](https://www.gatsby.ucl.ac.uk/~rapela) | practical |
    | 02 | Jan 20 | [Statistics for temporal time series analysis](https://github.com/joacorapela/statNeuro2025/blob/master/lectures/02_temporalTimeSeriesAnalysis/temporalTimeSeriesAnalysis.pdf)| [Joaquin Rapela](https://www.gatsby.ucl.ac.uk/~rapela) | lecture |
    | 02 | Jan 24 | Statistics for temporal time series analysis | [Joaquin Rapela](https://www.gatsby.ucl.ac.uk/~rapela) | practical |
    | 03 | Jan 27 | [Spectral time series analysis](https://github.com/joacorapela/statNeuro2025/blob/master/lectures/03_spectralTimeSeriesAnalysis/spectralTimeSeriesAnalysis.pdf) | [Joaquin Rapela](https://www.gatsby.ucl.ac.uk/~rapela) | lecture |
    | 03 | Jan 31 | Spectral time series analysis | [Joaquin Rapela](https://www.gatsby.ucl.ac.uk/~rapela) | practical |
    | 04 | Feb 03 | [Dimensionality reduction](https://github.com/joacorapela/statNeuro2025/blob/master/lectures/04_dimensionalityReduction/dimension-reduction-pca.pdf) | [Sina Tootoonian](https://sinatootoonian.com) | lecture |
    | 04 | Feb 07 | [Dimensionality reduction](https://github.com/joacorapela/statNeuro2025/tree/master/practicals/04_dimensionalityReduction) | [Sina Tootoonian](https://www.linkedin.com/in/sina-tootoonian-99668838/) | practical |
    | 05 | Feb 10 | [Linear Regression](https://github.com/joacorapela/neuroinformatics24/blob/master/lectures/06_linearRegression/swc_neuroinformatics_linreg.pdf) | [Lior Fox](https://liorfox.github.io/) | lecture |
    | 05 | Feb 14 | [Linear Regression](https://github.com/joacorapela/statNeuro2025/tree/master/practicals/05_linearRegression) | [Lior Fox](https://liorfox.github.io/) | practical |
    | 06 | Feb 17 | [Circular (directional) statistics](https://github.com/joacorapela/statNeuro2025/blob/master/lectures/06_cicularStatistics/circular_statistics.pdf) | [Kira D&#252;sterwald](https://scholar.google.com/citations?user=U7NxV-MAAAAJ&hl=en) | lecture |
    | 06 | Feb 21 | [Circular (directional) statistics](https://github.com/joacorapela/statNeuro2025/blob/master/practicals/06_circularStatistics/directional_statistics_practical.ipynb) | [Kira D&#252;sterwald](https://scholar.google.com/citations?user=U7NxV-MAAAAJ&hl=en) | practical |
    | 07 | Feb 24 | [Linear Dynamical Systems](lectures/07_linearDynamicalSytems/LDS_Jensen_2025.pdf) | [Kris Jensen](https://krisjensen.github.io/) | lecture |
    | 07 | Feb 28 | [Linear Dynamical Systems](https://colab.research.google.com/drive/1kVDxgEw_aG9HgVp7-5OzE-OPzCtw1dov?usp=sharing) | [Kris Jensen](https://krisjensen.github.io/) | practical |
    | 08 | Mar 03 | [Artificial Neural Networks](https://slides.com/eringrant/2024-03-07-swc-neural-nets-lecture/fullscreen?token=Gq60IrMy) | [Erin Grant](https://eringrant.github.io/) | lecture |
    | 08 | Mar 07 | [Artificial Neural Networks](https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html) | [Erin Grant](https://eringrant.github.io/) | practical |
    | 09 | Mar 10 | [Reinforcement learning](lectures/10_reinforcementLearning/RLinTheBrain_SWC_2024.pdf) | [Jesse Geerts](https://scholar.google.com/citations?user=4xusDVAAAAAJ&hl=en) | lecture |
    | 09 | Mar 14 | [Reinforcement learning](https://github.com/joacorapela/statNeuro2025/tree/master/practicals/09_reinforcementLearning) | [Jesse Geerts](https://scholar.google.com/citations?user=4xusDVAAAAAJ&hl=en) | practical |
    | 10 | Mar 17 | [Experimental Control with Bonsai](https://neurogears.org/neuroinformatics-2024/) | [Goncalo Lopez](https://neurogears.org/about-us/) [Nick Guilbeault](https://www.linkedin.com/in/ncguilbeault/) | lecture |
    | 10 | Mar 21 | Experimental Control with Bonsai | [Goncalo Lopez](https://neurogears.org/about-us/) [Nick Guilbeault](https://www.linkedin.com/in/ncguilbeault/) | practical |

- running scripts: we recommend that you run the provided scripts in a conda environment. Before running any script do (only once):

    1. `conda create -n statsNeuro python`
    2. clone this repository (`git clone git@github.com:joacorapela/statsNeuro2025.git`)

    3. change to the repository directory (`cd statsNeuro2025`)
    4. activate your conda environment (`conda activate statsNeuro`)
    5. type `pip install -e .` to install requirements

    Then you can run any script by (for example):

    - `cd practicals/02_LFPs_spectralAnalysis/exercises/`
    - `python doReconstructionExercise.py`

- If you have any problem running scripts in this repo, please contact [Joaquin Rapela](https://www.gatsby.ucl.ac.uk/~rapela).

