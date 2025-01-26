# Overview

This project focuses on predicting whether a gene and its associated enhancer form a regulatory pair. Gene-enhancer regulation plays a critical role in controlling gene expression, with significant implications for understanding cellular processes, development, and disease mechanisms. Using machine learning and deep learning techniques, this project builds models to predict the likelihood of regulation between gene-enhancer pairs based on genomic and epigenetic features.

## Key Components of the Project

1. **Data Collection and Processing**
   - Compiled gene-enhancer data from publicly available genomic datasets.
   - Preprocessed the data to extract relevant features, such as:
     - Genomic distances.
     - Chromatin interaction metrics.
     - Enhancer activity and gene expression correlations.
     - Epigenetic markers, including histone modifications and DNA accessibility.

2. **Feature Engineering**
   - Designed features to capture biological relationships between genes and enhancers, focusing on genomic and epigenetic contexts.
   - Incorporated spatial and sequence-based characteristics to enhance prediction accuracy.

3. **Model Development**
   - Tested and compared the performance of various models:
     - **Logistic Regression**: Established as the baseline for performance comparison.
     - **Convolutional Neural Networks (CNNs)**: Utilized to learn spatial and interaction patterns from the data.
     - **XGBoost**: Applied for its ability to handle complex feature interactions and deliver high-performance predictions.

4. **Evaluation Methodology**
   - Used **Hold-One-Chromosome-Out Cross-Validation** as the primary evaluation metric:
     - For each chromosome, trained models on all other chromosomes and tested them exclusively on the held-out chromosome.
     - This approach ensured robust evaluation by mimicking real-world scenarios where regulatory patterns may vary across chromosomes.

This project highlights the synergy between machine learning, deep learning, and domain knowledge in addressing complex biological questions, bridging computational predictions with experimental insights.

# Contributors
1. Ricky Miura
2. Saachi Shenoy
3. Manav Dixit
4. Tonoya Ahmed
5. Aishani Mohapatra