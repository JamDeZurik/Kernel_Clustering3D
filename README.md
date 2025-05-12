# Wheat Kernel Clustering 3D
A project for Machine Learning Spring 2025. This project contains three files: _Seed_Data.csv_, _DBSCANViz.py,_ and _Clustering.py_.

**Seed_Data.csv**: the dataset taken from the [Wheat Seeds UCI](https://www.kaggle.com/datasets/dongeorge/seed-from-uci) repository. It contains information about a kernel's width, height, ratio, compactness, and more; however, this project only uses the first three.

**DBSCANViz.py**: an additional file for testing with the DBSCAN algorithm.

**Clustering.py**: the program which runs KMeans clustering in three dimensions for better accuracy. 
Precisely, 2D gave an adjusted random score (ARI) of 0.54 while 3D gives 0.73, which is a 35% increase!

Below is a plot of the data.
![image](https://github.com/user-attachments/assets/59d285ad-2ded-425e-bcc7-c6e2cef87e03)
