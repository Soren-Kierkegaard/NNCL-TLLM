# Time Series Forecasting with LLMs via Nearest Neighbor Contrastive Learning



|   This is under developpment  |

|_______________________________|

This an implementation attemped of the paper "Rethinking Time Series Forecasting with LLMs via Nearest Neighbor Contrastive Learning" from (https://arxiv.org/pdf/2412.04806)

# Répertoire

## Repository Files:
```
├── REAMDE.md
├── tsctp.py
└── To come
    └── ...
```

# Architecture

## Time Series Compatible Text Prototypes (TCTPs) :
  
<img width="640" height="265" alt="nncl-tllm" src="https://github.com/user-attachments/assets/f2530e49-0fcb-492f-97a1-444fe76bc649" />

- Core Approach:

  Instead of directly mapping time series to word tokens, the method creates learnable "text prototypes":

- But what does that mean ?
  
  The goal is to try to align time series representations with text to make the input comprehensible to LLMs.
  Each prototype represents nearby word tokens in the embedding space, such as each text prototype represents word token embeddings (in its neighborhood) + time series characteristics
  and are learned to be "neighborhood-aware".


Nearest Neighbor Contrastive Learning: Borrowed from computer vision, this technique:

Maintains a support set (queue) of TCTPs

For each time series input, finds the top-k nearest neighbor TCTPs

Uses contrastive loss to align time series embeddings with appropriate text prototypes

Formulates prompts by concatenating time series patch embeddings with nearest neighbor TCTPs

  
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
