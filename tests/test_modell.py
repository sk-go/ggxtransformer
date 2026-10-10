import copy

import pytest
import torch

from chroma.modell import ChromaConfig, ChromaModell


def klein(farbe):
    torch.manual_seed(0)
    cfg = ChromaConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=8, farbe=farbe)
    m = ChromaModell(cfg).double().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.5)  # große Gewichte, damit die Nichtlinearitäten sichtbar wirken
    return m


def transformiere(model, M):
    """Wendet die orthogonale Matrix M auf alle Gewichte an, die den Stream lesen oder schreiben."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.wte.weight.copy_(m.wte.weight @ M)
        m.wpe.weight.copy_(m.wpe.weight @ M)
        for b in m.h:
            for lin in (b.attn.c_attn, b.mlp.c_fc):
                lin.weight.copy_(lin.weight @ M)
            for lin in (b.attn.c_proj, b.mlp.c_proj):
                lin.weight.copy_(M.T @ lin.weight)
                lin.bias.copy_(lin.bias @ M)
        m.lm_head.weight.copy_(m.lm_head.weight @ M)
    return m


def drehung(d, seed=1):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def permutation(d, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.eye(d, dtype=torch.float64)[torch.randperm(d, generator=g)]


IDX = torch.tensor([[1, 5, 7, 3, 9, 2, 4, 8]])


def test_logits_form_und_entkopplung():
    m = klein("keine")
    assert m(IDX).shape == (1, 8, 50)
    assert m.lm_head.weight.data_ptr() != m.wte.weight.data_ptr()


def test_alpha_archetyp_nichtnegativ():
    assert (klein("alpha").archetyp() >= 0).all()


def test_keine_farbe_ist_drehinvariant():
    m = klein("keine")
    assert torch.allclose(m(IDX), transformiere(m, drehung(8))(IDX), atol=1e-9)


@pytest.mark.parametrize("farbe", ["alpha", "beta"])
def test_farbe_bricht_drehung(farbe):
    m = klein(farbe)
    assert (m(IDX) - transformiere(m, drehung(8))(IDX)).abs().max() > 1e-3


@pytest.mark.parametrize("farbe", ["keine", "alpha", "beta"])
def test_alle_sind_permutationsinvariant(farbe):
    m = klein(farbe)
    assert torch.allclose(m(IDX), transformiere(m, permutation(8))(IDX), atol=1e-9)


def test_strom_laenge():
    assert len(klein("beta").strom(IDX)) == 3  # Eingang Block 0, Eingang Block 1, Ausgang


def test_falsche_farbe():
    with pytest.raises(ValueError):
        ChromaConfig(farbe="rot")


def test_zu_langer_kontext():
    with pytest.raises(ValueError):
        klein("keine")(torch.zeros(1, 17, dtype=torch.long))
