import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Optional

"""
     _______             ____        _           _____                     __  _ __   __      ______        __    ___           __       __                  
    /_  __(_)_ _  ___   / __/__ ____(_)__ ___   / ___/__  __ _  ___  ___ _/ /_(_) /  / /__   /_  __/____ __/ /_  / _ \_______  / /____  / /___ _____  ___ ___
     / / / /  ' \/ -_) _\ \/ -_) __/ / -_|_-<  / /__/ _ \/  ' \/ _ \/ _ `/ __/ / _ \/ / -_)   / / / -_) \ / __/ / ___/ __/ _ \/ __/ _ \/ __/ // / _ \/ -_|_-<
    /_/ /_/_/_/_/\__/ /___/\__/_/ /_/\__/___/  \___/\___/_/_/_/ .__/\_,_/\__/_/_.__/_/\__/   /_/  \__/_\_\\__/ /_/  /_/  \___/\__/\___/\__/\_, / .__/\__/___/.
                                                             /_/                                                                          /___/_/    
"""

class TCTPLearner(nn.Module):
    """
    Time Series Compatible Text Prototype (TCTP) Learner
    
    Apprend des prototypes de texte compatibles avec les séries temporelles
    en utilisant les word token embeddings d'un LLM.
    """
    
    def __init__(
        self,
        word_token_embeddings: torch.Tensor,
        num_prototypes: int = 1000,
        embedding_dim: int = 768,
        support_set_size: int = 10000,
        top_k: int = 8,
        temperature: float = 0.07
    ):
        """
        Args:
            word_token_embeddings: Embeddings des word tokens du LLM (V x D)
            num_prototypes: Nombre de TCTPs à apprendre (U)
            embedding_dim: Dimension des embeddings (D)
            support_set_size: Taille de la queue de support (q)
            top_k: Nombre de voisins les plus proches à utiliser
            temperature: Température pour la loss contrastive
            
        Return(s):
            _
        """
        super().__init__()
        
        # Embeddings des word tokens (figés)
        self.register_buffer('word_embeddings', word_token_embeddings)
        self.vocab_size = word_token_embeddings.shape[0]
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.top_k = top_k
        self.temperature = temperature
        
        # TCTPs apprenables (initialisés aléatoirement)
        self.tctp_embeddings = nn.Parameter(
            torch.randn(num_prototypes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.tctp_embeddings)
        
        # Support set (queue FIFO) pour NNCL
        self.support_set_size = support_set_size
        self.support_queue = deque(maxlen=support_set_size)
        
    def compute_prototype_loss(self) -> torch.Tensor:
        """
        Calcule L_proto : MSE entre chaque word token et son TCTP le plus proche
        
        Returns:
            Loss prototype (scalaire)
        """
        # Normalisation L2 pour stabilité
        word_emb_norm = F.normalize(self.word_embeddings, p=2, dim=1)
        tctp_norm = F.normalize(self.tctp_embeddings, p=2, dim=1)
        
        # Calcul des distances euclidiennes (V x U)
        distances = torch.cdist(word_emb_norm, tctp_norm, p=2)
        
        # Trouver le TCTP le plus proche pour chaque word token
        min_distances, nearest_indices = torch.min(distances, dim=1)
        nearest_prototypes = tctp_norm[nearest_indices]
        
        # MSE loss
        loss = F.mse_loss(word_emb_norm, nearest_prototypes)
        
        return loss
    
    def update_support_set(self, batch_tctps: torch.Tensor):
        """
        Met à jour le support set (queue FIFO) avec les TCTPs du batch actuel
        
        Args:
            batch_tctps: Embeddings TCTP du batch (B x D ou U x D)
        """
        # Détacher pour éviter l'accumulation de gradients
        batch_tctps_detached = batch_tctps.detach().cpu()
        
        # Ajouter à la queue (supprime automatiquement les plus anciens)
        for tctp in batch_tctps_detached:
            self.support_queue.append(tctp)
    
    def get_nearest_neighbor_tctps(self, time_series_embedding: torch.Tensor) -> torch.Tensor:
    
        """
        Obtient les top-k TCTPs les plus proches du support set
        
        Args:
            time_series_embedding: Embedding de la série temporelle (B x D)
            
        Returns:
            Top-k TCTPs les plus proches (B x k x D)
        """
        
        if len(self.support_queue) == 0:
            # Initialisation : utiliser les TCTPs actuels
            return self.tctp_embeddings[:self.top_k].unsqueeze(0).repeat(
                time_series_embedding.shape[0], 1, 1
            )
        
        # Convertir la queue en tensor
        support_set = torch.stack(list(self.support_queue)).to(
            time_series_embedding.device
        )  # (q x D)
        
        # Normalisation L2
        ts_emb_norm = F.normalize(time_series_embedding, p=2, dim=1)  # (B x D)
        support_norm = F.normalize(support_set, p=2, dim=1)  # (q x D)
        
        # Calcul des distances pour chaque élément du batch
        batch_size = ts_emb_norm.shape[0]
        nearest_tctps = []
        
        for i in range(batch_size):
            # Distances euclidiennes (q,)
            distances = torch.norm(
                support_norm - ts_emb_norm[i].unsqueeze(0), 
                dim=1, 
                p=2
            )
            
            # Top-k indices
            _, topk_indices = torch.topk(
                distances, 
                k=min(self.top_k, len(distances)), 
                largest=False
            )
            
            # Récupérer les TCTPs correspondants
            nearest_tctps.append(support_norm[topk_indices])
        
        return torch.stack(nearest_tctps)  # (B x k x D)
    
    def compute_nncl_loss(
        self,
        time_series_embedding: torch.Tensor,
        nearest_tctps: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la loss de contrastive learning avec nearest neighbors
        
        Args:
            time_series_embedding: Embedding de la série temporelle (B x D)
            nearest_tctps: Top-k TCTPs les plus proches (B x k x D)
            
            Calcule (simplification des calculs): 
                cos_sim = (A · B) / (||A|| × ||B||)
            
                To L2 Normalisation => A_norm = A / ||A||  # ||A_norm|| = 1
                                       B_norm = B / ||B||  # ||B_norm|| = 1
                                       
                                    => cos_sim = A_norm · B_norm
            
        Returns:
            Loss NNCL (scalaire)
        """
        
        batch_size = time_series_embedding.shape[0]
        
        # ================ Etape 1 : Normalisation L2
        # Après cette étape tout les vecteurs on une norme = 1
        ts_emb_norm = F.normalize(time_series_embedding, p=2, dim=1)  # (B x D)
        
        # =============== Etape 2 : SIMILARITÉ POSITIVE
        # Moyenne des top-k voisins comme positifs
        positive = nearest_tctps.mean(dim=1)  # (B x D)
        positive_norm = F.normalize(positive, p=2, dim=1)
        
        # Similarité positive
        pos_sim = torch.sum(ts_emb_norm * positive_norm, dim=1) / self.temperature  # (B,)
        
        # =============== Etape 2 : SIMILARITÉ Négative
        # Similarités négatives (entre éléments du batch)
        neg_sim = torch.mm(ts_emb_norm, ts_emb_norm.t()) / self.temperature  # (B x B)
        
        # Masquer la diagonale (similarité avec soi-même)
        mask = torch.eye(batch_size, device=neg_sim.device).bool()
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        
        # =============== Etape 4 : LOSS CONTRASTIVE
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B x B+1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # Calculer cross entropy logits vs labels
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def forward(
        self, 
        time_series_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass complet
        
        Args:
            time_series_embedding: Embedding de la série temporelle (B x D)
            
        Returns:
            - nearest_tctps: Top-k TCTPs pour formuler le prompt (B x k x D)
            - loss_proto: Loss prototype
            - loss_nncl: Loss contrastive
        """
        # 1. Calcul de la loss prototype
        loss_proto = self.compute_prototype_loss()
        
        # 2. Obtenir les nearest neighbor TCTPs
        nearest_tctps = self.get_nearest_neighbor_tctps(time_series_embedding)
        
        # 3. Calcul de la loss NNCL
        loss_nncl = self.compute_nncl_loss(time_series_embedding, nearest_tctps)
        
        # 4. Mettre à jour le support set
        self.update_support_set(self.tctp_embeddings)
        
        return nearest_tctps, loss_proto, loss_nncl


# Version test sans LLM
if __name__ == "__main__":

    # Simuler les word token embeddings d'un LLM (ex: GPT-2)
    vocab_size = 50257  # Taille du vocabulaire GPT-2
    embedding_dim = 768  # Dimension des embeddings GPT-2
    
    # Word embeddings simulés (normalement chargés depuis le LLM) de taille (vocabulaire x embedding_size)
    word_embeddings = torch.randn(vocab_size, embedding_dim)
    
    # Initialiser le TCTP Learner
    tctp_learner = TCTPLearner(
        word_token_embeddings = word_embeddings,
        num_prototypes = 1000,
        embedding_dim = 768,
        support_set_size = 10000,
        top_k = 8,
        temperature = 0.07
    )
    
    # Simuler un batch de time series embeddings
    batch_size = 16
    ts_embeddings = torch.randn(batch_size, embedding_dim)
    
    # Forward pass
    nearest_tctps, loss_proto, loss_nncl = tctp_learner(ts_embeddings)
    
    print(f"Shape des nearest TCTPs: {nearest_tctps.shape}")  # (16, 8, 768)
    print(f"Loss prototype: {loss_proto.item():.4f}")
    print(f"Loss NNCL: {loss_nncl.item():.4f}")
    print(f"Taille du support set: {len(tctp_learner.support_queue)}")