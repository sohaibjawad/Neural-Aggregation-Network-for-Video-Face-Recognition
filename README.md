# Neural-Aggregation-Network-for-Video-Face-Recognition
This repo is an implementation of 'Neural Aggregation Network for Video Face Recognition' by Jiaolong Yang. The basic idea behind it is to represent multiple features of the same subject as a single aggregated feature. While aggregation images which are clearer gets a higher weight as compared to ones that are blur. A model is trained for this purpose. <br>

<strong>crop_faces.py</strong> crops faces from the dataset. I used YTF dataset <br>
<strong>extract_features.py</strong> extracts feature embeddings for each face image and generates a dictionary where key is subject name and value is a list of embeddings <br>
<strong>data.py</strong> loads and sends data from dictionary in batches to model.py <br>
<strong>model.py</strong> trains a simple aggregation module <br>
<strong>single_feature_embeddings.py</strong> aggregates features of all subjects <br>
<strong>benchmark.py</strong> evaluates this technique on YTF dataset <br>
