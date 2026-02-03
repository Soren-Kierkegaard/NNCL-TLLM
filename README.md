# Time Series Forecasting with LLMs via Nearest Neighbor Contrastive Learning

 _______________________________
|                               |
|   This is under developpment  |
|_______________________________|

This an implementation attemped of the paper "Rethinking Time Series Forecasting with LLMs via Nearest Neighbor Contrastive Learning" from (https://arxiv.org/pdf/2412.04806)

# Répertoire

## Repository Files:
```
├── REAMDE.md
├── main.py
├── opti
    └── search_params.py     # Contain func for params optimization
├── pipeline
    ├── nncltllm.py          # NNCLTLLM complete model
    ├── train_eval.py        # Complete Train pipeline
    └── modules
        ├── normalisation.py  # Part of NNCLTLLM
        ├── patchembedding.py # same
        └── tsctp.py          # same
├── tools
    └── timeseries_dataset.py # Time serie representatin
├── viz
    └── visualization         # Tools for viz 
└── config.json               # Input config file
```

# Architecture

## Time Series Compatible Text Prototypes (TCTPs) :
  
<img width="640" height="265" alt="nncl-tllm" src="https://github.com/user-attachments/assets/f2530e49-0fcb-492f-97a1-444fe76bc649" />

- The Fundamental Problem 

  LLMs (such as GPT-2) are trained on text. Their "knowledge" is encoded in the space of word embeddings. 
  But how can we make a model that only "speaks" the language of text understand time series (numbers)? 
  Solution might be to either:
  
  1. Convert numbers into words: [1.5, 2.3, 1.8] → "one point five, two point three, one point eight".
  2. Or to Tokenize directly the numerical values
  
  The problem remains. Those 2 approach doesn't capture the temporal semantic of the time serie such as: seasonality, pattern, trend, and so for. 
  
  ==> <b>Instead of directly mapping time series to word tokens, the method creates learnable "text prototypes"</b> and create a breach between to modalities:

      Texte ←→ [TCTPs] ←→ TS

- But what does that mean ?
  
  The goal is to try to align time series representations with text to make the input comprehensible to LLMs.
  Each prototype represents nearby word tokens in the embedding space, such as each text prototype represents word token embeddings (in its neighborhood) + time series characteristics
  and are learned to be "neighborhood-aware".

  So a <b>Text Prototype</b> is a vector in the embedding space wich:

   * Live in the same vector space that embbedings words of the LLM
   * But does not correspond to any real world word, it more like a neologism or pseudo word compose of different unit language, just like sampling the concept of "cat ears" in the image space can lead to have non realistic ears for cats but imaginable regarding the distribution values of cat ears representation
   * It's "pseudo-word", learned to represent concepts for time series

    <figure>
        <img width="890" height="375" alt="emb_series" src="https://github.com/user-attachments/assets/69aed6fa-2c5b-41d7-abd0-0616b41bc3ed" />
        <br/>
        <figcaption><font color="#9900FF">TCTP (Time series Compatible Text Prototype) are pseudo-words vector representation that best capture time series semantic concept</font></figcaption>
    </figure>
 

  ### Steps:

     1. The word embedding space is take from any know LLM such as GPT or Ollama model, for example GPT-2 has a vocabulary of ~50K words each are represented by an 768 embedded vector

      "temperature" → [0.23, -0.45, 0.12, ..., 0.67]  (768 valeurs)
      "increase"    → [0.18, 0.32, -0.08, ..., 0.45]
      "pattern"     → [-0.12, 0.56, 0.34, ..., -0.23]
       ....

     2. Create the TCTP by initialising a 1000 learnable vectors of the same embedding size that are randomly initialize
        
       TCTP_1 → [0.15, -0.22, 0.08, ..., 0.34]
       TCTP_2 → [0.42, 0.11, -0.15, ..., 0.56]
                       ...
       TCTP_1000 → [-0.08, 0.33, 0.21, ..., -0.12]

     3. Learning the "Neighborhood-Aware" where each TCTP must represent a group of similar words

        In the artice, they use a "L proto" or a loss prototype which aims to maximise distance between non-similar pair and minimise it between similar pairs.
        Similar approach can be find in what we call "Siamese Networks" (https://youtu.be/6jfw8MuKwpI).

        ```
        # For each word of the vocab ....
        for word_embedding in vocabulary:
            # ... find the closest TCTP, here we can use cosinus similarity
            nearest_tctp = find_closest(word_embedding, all_tctps)
            
            # ... compute the loss
            loss += distance(word_embedding, nearest_tctp)²
        ```

        The result is that TCTPs are startegicly positioned in the sapce that "cover" the word close to the concept they want to represent.

    4. <b>Neirest Neighbor Contrastive Learning (NNCL)</b> - Make the TCTPs Time Series Compatible
 
       The next step is to match some of those TCTPs with our time series (because for now our TCTPs know sapce words but not time series yet)
 
       The loss is describe by the formula:

           L_NNCL = -log( exp(sim(anchor, positive) / τ) / Σ exp(sim(anchor, all_samples) / τ) )

       if we try to decompose it: 

           ```
            python#
            1. Similaritities with positives (TCTPs close)
            positive_sim = cosine_similarity(ts_embedding, mean(nearest_tctps))
            numerator = exp(positive_sim / temperature)
            
            # 2. Similarities with ALL series from the batch (negatives)
            all_sims = [cosine_similarity(ts_embedding, other_ts) 
                        for other_ts in batch]
            denominator = sum(exp(sim / temperature) for sim in all_sims)
            
            # 3. Loss finale
            loss = -log(numerator / denominator)
           ```
   
- Nearest Neighbor Contrastive Learning

  - As previously mentioned we use Contrative Loss to learn to separate stuff by making those similar closest to one another and the other very distant.
    The difference between classical Contrastive Learning and Nearest Neighbor Contrastive Learning is that we use the K-closest neihgboor whereas we use a single positiv/negativ in the other.

    Classique :

        anchor ←→ 1 positive
        # Loss contrastive
        loss = distance(anchor, positive)²  # Minimise
                - distance(anchor, negative)²  # Maximise

    Nearest Neighbor :

        anchor ←→ k positives (K-closest)

This makes learning more robust, as several positive examples are used instead of just one.

<ins>Note:</ins> The anchor is a specific example of data that serves as a reference for comparing other examples (but it not a ground truth - not necessary the reality of the field but rather a starting point for comparison)


_____________________________________________________

# Dataset

- 

<table>
  <tr>
    <th>  </th>
    <th>  </th>
  </tr>
</table>

# Acknowlege

- Octobre 2025
