# Sense-Spectrum
These are the codes of the Sense Spectrum model proposed in our IJCNN 2022 paper [Dense Embeddings Preserving the Semantic Relationships in WordNet](https://ieeexplore.ieee.org/abstract/document/9892238) (In case one cannot have access to the official paper, the pre-print version is available [here](https://arxiv.org/pdf/2004.10863.pdf)).

In this model, we provided low-dimensional dense embeddings for WordNet noun and verb synsets, where the hypernym-hyponym relationship between two synsets can be preserved in the embedding. Since our synset embedding operates like a spectrum, we call our embedding system Sense Spectrum. We have to specify that this model can only provide the spectra (embeddings) for **noun** and **verb** synsets in WordNet. At this moment, we are not able to embed adjective or adverb synsets, since there is no hypernym-hyponym relationship among them.

To implement our model, one may follow these steps:

1. Clone this repository via `git clone https://github.com/canlinzhang/Sense-Spectrum.git` to your local file.

2. Train the sense spectrum for noun and verb synsets: `python train_sense_spectra.py`. Only CPU is required for the training. On a Macbook Pro with Apple M1 chip, it takes around 7 hours for training. Also, one can directly download the trained noun and verb spectra via [this link](https://drive.google.com/drive/folders/17FwCkN1YU-Peig3sqBG9FuU4HB9sT538). After downloading, please put the two pickle files into the `synset_spactra` folder.

3. Evaluate the Hypernym Intersection Similarity (HIS) measurement using the SimLex-999 dataset: `python evaluate_HIS_dist_on_SimLex-999.py`. The HIS measurement out-performs the traditional synset similarity measurements of WordNet by achieving higher correlation covariance score on the SimLex-999 noun and verb word pairs. One may read our [paper](https://arxiv.org/pdf/2004.10863.pdf) for more details.

Finally, one can obtain the noun and verb spectrum for a given word. For example, we can obtain the noun and verb synset spectrum for the word 'dog' by implementing the following script:
```
from obtain_sense_spectra import ObtainSpectrum
Class = ObtainSpectrum('synset_spactra/noun_synset_spectra.pickle', 'synset_spactra/verb_synset_spectra.pickle')

noun_spec, verb_spec = Class.obtain_spectrum_for_word('dog')
```

Here, `noun_spec` is the average of the noun synset spectra related to 'dog'. To be specific, by evaluating `nltk wordnet`, we can see the noun synset related to 'dog' are `dog.n.01`, `frump.n.01`, `dog.n.03`, `cad.n.01`, `frank.n.02`, `pawl.n.01` and `andiron.n.01`. Then, the above script will obtain the spectrum for each of these synsets, average the obtained spectra dimension-wise, and then output the average spectra as `noun_spec`. The `verb_spec` is obtained similarly from the verb synsets related to 'dog', which is only `chase.v.01`.

Besides, if one already gets the synset (say, 'dog.n.01') and wants to obtain the corresponding spectrum, simply implement the same class as above and run
```
spec = Class.obtain_spectrum_for_synset('dog.n.01')
```

To cite our paper, please use:

C. Zhang and X. Liu, "Dense Embeddings Preserving the Semantic Relationships in WordNet," 2022 International Joint Conference on Neural Networks (IJCNN), Padua, Italy, 2022, pp. 01-08, doi: 10.1109/IJCNN55064.2022.9892238.
