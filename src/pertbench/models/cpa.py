import torch
import torch.nn as nn
import torch.nn.functional as F

from pertbench.config import ModelConfig


class Discriminator(nn.Module):
    """Predicts the multi-hot perturbation vector from the basal latent vector."""
    def __init__(self, latent_dim: int, n_perts: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_perts),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the logits for each perturbation."""
        return self.net(z)


class CPAModule(nn.Module):
    """Conditional VAE with an additive perturbation embedding.

    Encoder: x -> (mu, logvar). Sample z_basal. Add the perturbation
    embedding. 
    Decoder: z -> reconstructed x.
    """
    def __init__(self, n_genes: int, n_perts: int, latent_dim: int = 64, hidden_dim: int = 512) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        # No bias so the control vector (all zeros) maps to the zero embedding
        self.pert_embedding = nn.Linear(n_perts, latent_dim, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mu and logvar of the basal latent state."""
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample in training mode (reparameterization trick). Return mu in evaluation mode."""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_full(self, x: torch.Tensor, pert: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct x under the perturbation pert."""
        mu, logvar = self.encode(x)
        z_basal = self.sample(mu, logvar)
        z = z_basal + self.pert_embedding(pert)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z_basal

    def forward(
        self, x: torch.Tensor, pert: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as forward_full, without z_basal."""
        x_hat, mu, logvar, _ = self.forward_full(x, pert)
        return x_hat, mu, logvar

    @torch.no_grad()
    def predict_from_control(
        self, x_control: torch.Tensor, pert: torch.Tensor
    ) -> torch.Tensor:
        """Predict expression under pert, starting from control cells."""
        was_training = self.training
        self.eval()
        mu, _ = self.encode(x_control)
        x_hat = self.decoder(mu + self.pert_embedding(pert))
        self.train(was_training)
        return x_hat


def cpa_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute the VAE loss and its components."""
    recon_loss = ((x_hat - x) ** 2).sum(dim=1).mean()
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return {"total": recon_loss + kl_weight * kl_loss, "recon": recon_loss, "kl": kl_loss}


def disc_loss(logits: torch.Tensor, pert: torch.Tensor) -> torch.Tensor:
    """Compute binary cross-entropy against the true multi-hot vector."""
    bce = F.binary_cross_entropy_with_logits(logits, pert, reduction="none")
    return bce.sum(dim=1).mean()


def confusion_loss(logits: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy against the base rate of each perturbed gene."""
    target = prior.expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return bce.sum(dim=1).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig()
    n_genes, n_perts, batch = 2065, 102, 32

    model = CPAModule(n_genes, n_perts, cfg.latent_dim, cfg.hidden_dim)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    x = torch.randn(batch, n_genes)
    pert = torch.zeros(batch, n_perts)
    pert[:, 0] = 1.0
    x_hat, mu, logvar = model(x, pert)
    print("x_hat:", tuple(x_hat.shape), "| mu:", tuple(mu.shape), "| logvar:", tuple(logvar.shape))

    # Composition: the double embedding must equal the sum of the single embeddings
    a, b, ab = torch.zeros(n_perts), torch.zeros(n_perts), torch.zeros(n_perts)
    a[3], b[7] = 1.0, 1.0
    ab[3], ab[7] = 1.0, 1.0
    emb = model.pert_embedding
    print("double = sum of singles:", torch.allclose(emb(ab), emb(a) + emb(b), atol=1e-6))
    print("control embedding is zero:", bool((emb(torch.zeros(n_perts)) == 0).all()))

    # Evaluation mode is deterministic 
    # Training mode is random
    model.eval()
    out1, _, _ = model(x, pert)
    out2, _, _ = model(x, pert)
    print("eval mode deterministic:", torch.allclose(out1, out2))
    model.train()
    out3, _, _ = model(x, pert)
    out4, _, _ = model(x, pert)
    print("train mode random:", not torch.allclose(out3, out4))

    # predict_from_control must match eval-mode forward and restore the mode
    pred = model.predict_from_control(x, pert)
    model.eval()
    print("predict_from_control matches eval forward:", torch.allclose(pred, model(x, pert)[0]))
    model.train()
    model.predict_from_control(x, pert)
    print("training mode restored:", model.training)

    # Loss and gradients
    x_hat, mu, logvar = model(x, pert)
    loss = cpa_loss(x_hat, x, mu, logvar, cfg.kl_weight)
    print({k: round(float(v.detach()), 3) for k, v in loss.items()})
    loss["total"].backward()
    print("all parameters have gradients:", all(p.grad is not None for p in model.parameters()))
