"""typed run configuration for the IDEP runner.

Every hyperparameter of a run is explicit and validated at construction
(fail-fast, before any model build or data read): arm, dataset, backbone,
seed fraction/seed, epochs, batch, max_length, dropout, single Adam LR,
K (KS+KP), gamma/beta, pi/rho weight pairs (frozen defaults when absent),
device (explicit gpu index; -1 = auto-select least-loaded), the gating settings
(early-stop step/floor, engage steps), the first-step watchdog budget, and the
per-process CPU cap. JSON is the repo config format (the frozen screen configs
are JSON); this typed layer makes every field explicit and validated.
"""
from __future__ import annotations

import hashlib
import json

from common.data import CLASSES
from common.elhaj_data import ELHAJ_CLASSES

KS_DEFAULT = 12
KP_DEFAULT = 4


class ConfigError(ValueError):
    """A typed IDEP config violates the run contract."""


class IDEPRunConfig:
    """Immutable, validated configuration of one IDEP run.

    REQUIRED fields must be present in the config; OPTIONAL fields carry the
    frozen defaults (the frozen screen configs predate them and stay valid).
    """

    REQUIRED = ("arm", "run_name", "backbone", "seed_fraction", "epochs",
                "batch_size", "max_length", "dropout", "lr", "seed", "gpu")
    OPTIONAL = ("dataset", "k_singleton", "k_pair", "gamma", "beta",
                "lambda_kl", "lambda_m",
                "pi_weights", "rho_weights", "allow_cpu", "early_stop_step",
                "early_stop_floor", "engage_steps", "max_first_step_seconds",
                "cpu_cap_fraction")
    FIELDS = REQUIRED + OPTIONAL

    def __init__(self, d: dict) -> None:
        missing = [k for k in self.REQUIRED if k not in d]
        if missing:
            raise ConfigError(f"config missing fields: {missing}")
        unknown = [k for k in d if k not in self.FIELDS
                   and k not in ("weights_provenance", "_ablation", "_rescue",
                                 "_sweep", "_final")]
        if unknown:
            raise ConfigError(f"config unknown fields: {unknown}")

        self.arm = d["arm"]
        self.dataset = d.get("dataset", "madar")
        self.run_name = d["run_name"]
        self.backbone = d["backbone"]
        self.seed_fraction = int(d["seed_fraction"])
        self.epochs = int(d["epochs"])
        self.batch_size = int(d["batch_size"])
        self.max_length = int(d["max_length"])
        self.dropout = float(d["dropout"])
        self.lr = float(d["lr"])
        self.seed = int(d["seed"])
        self.gpu = int(d["gpu"])
        self.k_singleton = int(d.get("k_singleton", KS_DEFAULT))
        self.k_pair = int(d.get("k_pair", KP_DEFAULT))
        self.gamma = float(d.get("gamma", 1.0))
        self.beta = float(d.get("beta", 0.5))
        self.lambda_kl = float(d.get("lambda_kl", 0.5))
        self.lambda_m = float(d.get("lambda_m", 0.1))
        self.pi_weights = d.get("pi_weights")
        self.rho_weights = d.get("rho_weights")
        self.allow_cpu = bool(d.get("allow_cpu", False))
        self.early_stop_step = int(d.get("early_stop_step", 760))
        self.early_stop_floor = d.get("early_stop_floor", 0.8712)
        self.engage_steps = tuple(int(s) for s in d.get("engage_steps", (50, 152)))
        self.max_first_step_seconds = float(d.get("max_first_step_seconds", 600.0))
        self.cpu_cap_fraction = float(d.get("cpu_cap_fraction", 0.8))
        self.validate()

    # ------------------------------------------------------------- properties

    @property
    def classes(self) -> list:
        return ELHAJ_CLASSES if self.dataset == "elhaj" else CLASSES

    @property
    def class2idx(self) -> dict:
        return {c: i for i, c in enumerate(self.classes)}

    # ------------------------------------------------------------- validation

    def validate(self) -> None:
        if self.arm not in ("a0", "a2", "a3"):
            raise ConfigError(f"arm {self.arm!r} not in (a0, a2, a3)")
        if self.dataset not in ("madar", "elhaj"):
            raise ConfigError(f"dataset {self.dataset!r} not in (madar, elhaj)")
        if self.dataset == "elhaj" and self.k_pair != 0:
            raise ConfigError("ElHaj regime: all fibers singleton => k_pair must be 0")
        if self.seed_fraction not in (5, 10, 20):
            raise ConfigError(f"seed_fraction {self.seed_fraction} not in (5,10,20)")
        if self.epochs <= 0 or self.batch_size <= 0 or self.max_length <= 0:
            raise ConfigError("epochs/batch_size/max_length must be > 0")
        if not (0.0 < self.dropout < 1.0):
            raise ConfigError(f"dropout {self.dropout} not in (0,1)")
        if self.lr <= 0:
            raise ConfigError(f"lr {self.lr} <= 0")
        if self.seed < 0:
            raise ConfigError(f"seed {self.seed} < 0")
        if self.gpu < -1:
            raise ConfigError(f"gpu {self.gpu} < -1 (use -1 for auto-select)")
        if self.k_singleton < 0 or self.k_pair < 0:
            raise ConfigError("k_singleton/k_pair must be >= 0")
        if self.arm in ("a2", "a3") and self.k_singleton + self.k_pair == 0:
            raise ConfigError(f"arm {self.arm} requires K = KS+KP > 0")
        if not (0.0 <= self.gamma):
            raise ConfigError(f"gamma {self.gamma} < 0")
        if not (0.0 <= self.beta):
            raise ConfigError(f"beta {self.beta} < 0")
        if not (0.0 <= self.lambda_kl):
            raise ConfigError(f"lambda_kl {self.lambda_kl} < 0")
        if not (0.0 <= self.lambda_m):
            raise ConfigError(f"lambda_m {self.lambda_m} < 0")
        # anchor parity — frozen for every arm/dataset
        if self.lr != 3e-5:
            raise ConfigError(f"lr {self.lr} != frozen anchor 3e-5")
        if self.dropout != 0.5:
            raise ConfigError(f"dropout {self.dropout} != frozen anchor 0.5")
        if self.max_length != 128:
            raise ConfigError(f"max_length {self.max_length} != frozen anchor 128")
        if self.batch_size != 32:
            raise ConfigError(f"batch_size {self.batch_size} != frozen anchor 32")
        # weight pairs validate against the config's own class set
        if self.pi_weights is not None:
            _validate_weights(self.pi_weights, self.classes, "pi_weights")
        if self.rho_weights is not None:
            _validate_weights(self.rho_weights, self.classes, "rho_weights")
        if not (0.0 < self.cpu_cap_fraction <= 1.0):
            raise ConfigError(f"cpu_cap_fraction {self.cpu_cap_fraction} not in (0,1]")
        if self.max_first_step_seconds <= 0:
            raise ConfigError("max_first_step_seconds must be > 0")

    # ------------------------------------------------------------------- misc

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.FIELDS}

    def config_sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()


def _validate_weights(w: dict, classes: list, name: str) -> None:
    if not isinstance(w, dict) or set(w.keys()) != set(classes):
        raise ConfigError(f"{name}: keys {sorted(w) if isinstance(w, dict) else w!r} "
                          f"!= {sorted(classes)}")
    if any(v <= 0 for v in w.values()):
        raise ConfigError(f"{name}: all weights must be > 0")
    s = sum(w.values())
    if abs(s - 1.0) > 1e-6:
        raise ConfigError(f"{name}: weights sum {s!r} != 1 (+/- 1e-6)")


def load_config(path: str) -> IDEPRunConfig:
    with open(path) as f:
        return IDEPRunConfig(json.load(f))
