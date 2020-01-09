# Sense-Spectrum
 This is the code of our Sense Spectrum model proposed in our paper "Dense Embeddings Preserving the Semantic Relationships in WordNet".
 One may download all the files and put them in one folder. 
 Then, under this folder, please create two folders named 'checkpoints' and 'SimLex_999_spectrum'. 
 Then under the folder 'SimLex_999_spectrum', please create two folders named 'noun' and 'verb'. 
 After that, simply run the sense_spectrum.py will do. 
 The training takes about 20-30 hours, depending on the capacity of the GPU and CPU. 
 Results will appear automatically after training. 
 We records the 5 closest synsets for each noun and verb synsets. And we record the synset pair for each SimLex-999 noun and verb pair, as well as their initial HIS scalars and spectrum-based HIS scalars.
