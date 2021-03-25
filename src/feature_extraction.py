import json
import torch
import config
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from itertools import cycle
from sklearn.manifold import TSNE
from model import Code2Vec, Code2VecEncoder
from dataset import Code2VecDataset, Vocabulary, load_vocabularies

sns.set_style('darkgrid')
logging.basicConfig(level = logging.INFO)

def extract(model_path: str):
    word_vocab, path_vocab, label_vocab = load_vocabularies(f'./data/{config.DATASET}/{config.DATASET}.dict.c2v')
    encoder = Code2VecEncoder(len(word_vocab), len(path_vocab), config.EMBEDDING_DIM, config.CODE_VECTOR_DIM, config.DROPOUT)
    model = Code2Vec(encoder, config.TARGET_DIM).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    
    print(model)
    
    test_ds = Code2VecDataset(f'./data/{config.DATASET}/{config.DATASET}.test.c2v', word_vocab, path_vocab, label_vocab)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size = 1, num_workers = 1)
    num_samples = len(test_dataloader.dataset)
    logging.info(f'Loaded {num_samples} samples for feature extraction')

    with open(f'./data/{config.DATASET}/features.json', 'w') as writer:
        for sample in tqdm(test_dataloader, total=num_samples):
            label_id = sample[0].to(config.DEVICE).detach().to('cpu').numpy()[0]
            label = label_vocab.lookup_word(label_id)

            x_s, path, x_t = sample[1].to(config.DEVICE), sample[2].to(config.DEVICE), sample[3].to(config.DEVICE)
            ap, af = model.extract_features(x_s, path, x_t)
            cluster_label = torch.argmax(ap)

            writer.write(json.dumps({"label": label, "assignment_probability": ap.detach().to('cpu').numpy()[0].tolist(), "assignment_feature": af.detach().to('cpu').numpy()[0].tolist(), "cluster_label": cluster_label.detach().to('cpu').numpy().tolist()}) + "\n")

    visualize_clusters()

def visualize_clusters():
    
    df = pd.read_json(f'./data/{config.DATASET}/features.json', lines=True)
    classes = { 0: 'train', 1: 'save', 2: 'process', 3: 'forward', 4: 'predict'}
    k = len(classes)

    # Assign method name category
    df['method_name_category'] = df.label.map(lambda x: np.array([x.find(s) for s in classes.values()]).argmax())

    # Sample equal number of instances from each category to visualize
    df = df.groupby('method_name_category').apply(lambda grp: grp.sample(n=100)).reset_index(level=[0, 1], drop=True)

    # Use tSNE for visualizing clusters
    assignment_probabilities = list(df.assignment_probability.map(np.array))
    assignment_features = list(df.assignment_feature.map(np.array))

    tsne_ap = TSNE(n_components=3, verbose=1, perplexity=20, n_iter=3000)
    tsne_ap_results = tsne_ap.fit_transform(list(assignment_probabilities))

    tsne_af = TSNE(n_components=3, verbose=1, perplexity=20, n_iter=3000)
    tsne_af_results = tsne_af.fit_transform(assignment_features)

    print("Silhouette Coefficient for Assignment Probabilities: %0.3f"
            % metrics.silhouette_score(assignment_probabilities, df.cluster_label.values, metric='sqeuclidean'))

    print("Silhouette Coefficient for Assignment Features: %0.3f"
            % metrics.silhouette_score(assignment_features, df.cluster_label.values, metric='sqeuclidean'))


    

    fig = plt.figure(figsize=(14,10))
    fig.suptitle(f'Visualizing (k={k}) clusters with tSNE')
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_title('Assignment Probability Clusters')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for klass, color in zip(range(0, k), colors):
        Xk = tsne_ap_results[df['cluster_label'] == klass]
        ax1.scatter(Xk[:, 0], Xk[:, 1], Xk[:, 2], c=color, alpha=0.3, label=klass)

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.set_title('Method name as label')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for klass, color in zip(range(0, k), colors):
        Xk = tsne_ap_results[df['method_name_category'] == klass]
        ax2.scatter(Xk[:, 0], Xk[:, 1],  Xk[:, 2], c=color, alpha=0.3, label=classes[klass])
    ax2.legend()


    ax3 = fig.add_subplot(223, projection='3d')
    ax3.set_title('Assignment Feature Clusters')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for klass, color in zip(range(0, k), colors):
        Xk = tsne_af_results[df['cluster_label'] == klass]
        ax3.scatter(Xk[:, 0], Xk[:, 1], Xk[:, 2], c=color, alpha=0.3, label=klass)

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title('Method name as label')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for klass, color in zip(range(0, k), colors):
        Xk = tsne_af_results[df['method_name_category'] == klass]
        ax4.scatter(Xk[:, 0], Xk[:, 1],  Xk[:, 2], c=color, alpha=0.3, label=classes[klass])
    ax4.legend()

    print("Silhouette Coefficient for Assignment Probabilities: %0.3f"
            % metrics.silhouette_score(assignment_probabilities, df.cluster_label.values, metric='sqeuclidean'))

    print("Silhouette Coefficient for Assignment Features: %0.3f"
            % metrics.silhouette_score(assignment_features, df.cluster_label.values, metric='sqeuclidean'))

    plt.show()

if __name__ == '__main__':
    extract('./models/code2vec.ckpt')