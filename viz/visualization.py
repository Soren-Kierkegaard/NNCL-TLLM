from collections import deque
from typing import Tuple, Optional, List

def visualize_predictions(inputs: torch.Tensor, predictions: torch.Tensor, ground_truths: torch.Tensor, num_samples: int = 3, channel_names: Optional[List[str]] = None, save_path: str = 'predictions.png'):
    
    """
    Visualise les prédictions vs ground truth

    Args:
        inputs: (N, C, T) séries d'entrée
        predictions: (N, C, H) prédictions
        ground_truths: (N, C, H) vérités terrain
        num_samples: nombre d'échantillons à visualiser
        channel_names: noms des canaux
        save_path: chemin pour sauvegarder la figure
    """

    if len(ground_truths.shape) == 2:
      # Add dimension
      ground_truths = ground_truths.unsqueeze(2)

    num_channels = inputs.shape[1]
    #pred_length  = predictions.shape[1]
    print(f"inputs : {inputs} / {inputs.shape}")
    print(f"Nombre de canaux : {num_channels}")
    print(f"Taille prédictio : {pred_length}")
    print(f"Num channel : {num_channels}")

    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(num_channels)]

    # Sélectionner des échantillons aléatoires de taille num_samples a visualiser
    indices = np.random.choice(len(inputs), size=min(num_samples, len(inputs)), replace=False)

    fig = plt.figure(figsize=(15, 4 * num_samples))
    gs  = GridSpec(num_samples, num_channels, figure=fig, hspace=0.3, wspace=0.3)

    for sample_idx, idx in enumerate(indices):
        input_seq = inputs[idx].numpy()  # (C, T)
        pred_seq = predictions[idx].numpy()  # (C, H)
        true_seq = ground_truths[idx].numpy()  # (C, H)

        for channel_idx in range(num_channels):
            ax = fig.add_subplot(gs[sample_idx, channel_idx])

            # Longueurs des séquences
            input_len = input_seq.shape[1] # ,T
            pred_len  = pred_seq.shape[1]

            # Axe temporel
            input_time = np.arange(input_len)
            pred_time = np.arange(input_len, input_len + pred_len)

            # Tracer l'historique
            ax.plot(input_time, input_seq[channel_idx],
                   label='Historical', color='blue', linewidth=1.5, alpha=0.7)

            print(f"pred_seq: {pred_seq} / {pred_seq.shape}")

            # Tracer la prédiction
            if pred_length == 1:
                # Un seul point : utiliser scatter ou plot avec marker
                ax.scatter(pred_time, pred_seq[channel_idx],
                          label='Prediction', color='red', s=100, marker='*', zorder=5)
                ax.scatter(pred_time, true_seq[channel_idx],
                          label='Ground Truth', color='green', s=80, marker='o', zorder=5)

                # Ajouter valeur sur le point
                ax.text(x = pred_time, y = pred_seq[channel_idx] + 2, s = "%d" %pred_seq[channel_idx], ha="center") # décaler en haut le label de +25
                ax.text(x = pred_time, y = true_seq[channel_idx] + 2, s = "%d" %true_seq[channel_idx], ha="center") # décaler en haut le label de +25

            else:
                # Tracer la prédiction
                ax.plot(pred_time, pred_seq[channel_idx],
                      label='Prediction', color='red', linewidth=1.5, linestyle='--')

                # Tracer le ground truth
                ax.plot(pred_time, true_seq[channel_idx],
                      label='Ground Truth', color='green', linewidth=1.5, alpha=0.7)

            # Ligne verticale pour séparer historique et prédiction
            ax.axvline(x=input_len, color='gray', linestyle=':', alpha=0.5)

            # MSE pour ce canal
            mse = np.mean((pred_seq[channel_idx] - true_seq[channel_idx]) ** 2)
            mae = np.mean(np.abs(pred_seq[channel_idx] - true_seq[channel_idx]))

            print(f"channel_idx: {channel_idx}")
            ax.set_title(f'{channel_names[channel_idx]} - Sample {idx}\nMSE: {mse:.4f}, MAE: {mae:.4f}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('NNCL-TLLM: Time Series Forecasting Results', fontsize=16, y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualisation sauvegardée: {save_path}")
    plt.close()


def plot_training_history(history: dict, save_path: str = 'training_history.png'):
    """
    Visualise l'historique d'entraînement

    Args:
        history: dictionnaire avec les métriques par époque
        save_path: chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss totale
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Forecast Loss
    axes[0, 1].plot(epochs, history['train_forecast_loss'], 'g-', label='Train Forecast Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Forecast Loss (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Proto Loss
    axes[1, 0].plot(epochs, history['train_proto_loss'], 'r-', label='Train Proto Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Prototype Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # NNCL Loss
    axes[1, 1].plot(epochs, history['train_nncl_loss'], 'm-', label='Train NNCL Loss', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('NNCL Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Historique d'entraînement sauvegardé: {save_path}")
    plt.close()