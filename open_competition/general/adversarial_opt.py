# coding = 'utf-8'

from dataclasses import dataclass, field

"""
These are options for adversarial attacks based on gradient methods. 

The implemented methods include:

1. FGSM (Fast Gradient Sign Method, https://arxiv.org/abs/1412.6572).
2. FGM (Fast Gradient Method).
3. PGD (Projected Gradient Descent).
4. FreeAT (Free Adversarial Training).
5. YOPO (You Only Propagate Once).
6. FreeLB (Free Large-Batch).
7. SMART (Smoothness-inducing Adversarial Regularization).

"""


@dataclass
class AdversarialOptBase:
    pass


@dataclass
class FGSMOpt(AdversarialOptBase):
    eps: float = field(
        default=1e-5,
        metadata={'help':
                      "The noise coefficient to multiply the sign of gradient."
                      "Controls the extent of noise."}
    )

@dataclass
class FGMOpt(AdversarialOptBase):
    eps: float = field(
        default=1e-5,
        metadata={'help':
                      "The noise coefficient to multiply the sign of gradient divided by its norm."
                      "Controls the extent of noise."}
    )