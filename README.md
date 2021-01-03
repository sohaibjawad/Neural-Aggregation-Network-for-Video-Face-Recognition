# Neural-Aggregation-Network-for-Video-Face-Recognition
This repo is an implementation of 'Neural Aggregation Network for Video Face Recognition' by Jiaolong Yang. The basic idea behind it is to represent multiple features of the same subject as a single aggregated feature. While aggregation images which are clearer gets a higher weight as compared to ones that are blur. A model is trained for this purpose.

crop_faces.py crops faces from the dataset. I used YTF dataset
extract_features.py extracts feature embeddings for each face image and generates a dictionary where key is subject name and value is a list of embeddings
data.py loads and sends data from dictionary in batches to model.py
model.py trains a simple aggregation module
single_feature_embeddings.py aggregates features of all subjects
benchmark.py evaluates this technique on YTF dataset