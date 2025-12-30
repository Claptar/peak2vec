import torch
import torch.nn as nn
import torch.nn.functional as F


class Peak2Vec(nn.Module):
    """Skip-gram model over peaks with negative sampling."""

    def __init__(
        self,
        n_peaks: int,
        embedding_dim: int = 64,
        pos_weight: float = 1.0,
        sparse: bool = True,
        tie_weights: bool = False,
    ):
        super(Peak2Vec, self).__init__()
        self.dim = embedding_dim
        self.pos_weight = pos_weight
        self.tie_weights = tie_weights

        self.in_embedding = nn.Embedding(n_peaks, embedding_dim, sparse=sparse)
        self.out_embedding = (
            self.in_embedding
            if tie_weights
            else nn.Embedding(n_peaks, embedding_dim, sparse=sparse)
        )
        self.reset_params()

    def reset_params(self):
        nn.init.uniform_(self.in_embedding.weight, -0.5 / self.dim, 0.5 / self.dim)
        if not self.tie_weights:
            nn.init.uniform_(self.out_embedding.weight, -0.5 / self.dim, 0.5 / self.dim)

    def forward(
        self,
        peaks: torch.LongTensor,
        peak_pairs: torch.LongTensor,
        negatives: torch.LongTensor,
    ):
        """
        Compute the Skip-gram with Negative Sampling loss.
        peaks: (B,) LongTensor of peak indices
        peak_pairs: (B,) LongTensor of positive context peak indices
        negatives: (B, K) LongTensor of negative sample peak indices
        """
        # Embeddings
        peak_emb = self.in_embedding(peaks)  # (B, D)
        pair_emb = self.out_embedding(peak_pairs)  # (B, D)
        neg_emb = self.out_embedding(negatives)  # (B, K, D)

        # Compute similarity scores
        pos_score = torch.sum(peak_emb * pair_emb, dim=1)  # (B)
        neg_score = torch.bmm(neg_emb, peak_emb.unsqueeze(2)).squeeze(2)  # (B, K)

        # Loss
        poss_loss = F.softplus(-pos_score)  # -log(sigmoid(x)) = softplus(-x)
        neg_loss = F.softplus(neg_score).sum(1)  # -log(1 - sigmoid(x)) = softplus(x)
        loss = (self.pos_weight * poss_loss + neg_loss).mean()

        with torch.no_grad():
            stats = {
                "pos_score_mean": pos_score.mean().detach(),
                "neg_score_mean": neg_score.mean().detach(),
                "pos_loss_mean": poss_loss.mean().detach(),
                "neg_loss_mean": neg_loss.mean().detach(),
            }
        return loss, stats

    @torch.no_grad()
    def get_peak_embeddings(self, which: str = "in", normalize: bool = True):
        """
        Get peak embeddings.
        which (str):      Which embeddings to return: 'in', 'out', or 'avg'.
        normalize (bool): Whether to L2-normalize the embeddings.
        """
        if which == "in":
            emb = self.in_embedding.weight
        elif which == "out":
            emb = self.out_embedding.weight
        elif which == "avg":
            emb = 0.5 * (self.in_embedding.weight + self.out_embedding.weight)
        else:
            raise ValueError("which must be one of: 'in', 'out', 'avg'")

        embeddings = emb.detach().cpu()
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
