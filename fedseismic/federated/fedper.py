"""FedPer local trainer: shared base, private classifier.

Arivazhagan et al., arXiv:1912.00818. Aggregation stays plain averaging of the
uploaded state dicts. This module only decides which tensors are private.
"""

from copy import deepcopy

from .client import ClientTrainer


def personalization_keys(state_dict, prefix=None):
    """Names of the private head.

    Classification nets personalize ``classifier``. The seismic UNet
    personalizes ``outc`` and leaves the auxiliary ``reconstruct`` head in
    the shared base. Anything else personalizes the module of the last weight.
    """
    if prefix is None:
        names = list(state_dict)
        if any(name == "classifier.weight" or name.startswith("classifier.") for name in names):
            prefix = "classifier"
        elif any(name.startswith("outc.") for name in names):
            prefix = "outc"
        else:
            prefix = _last_weight_prefix(state_dict)
    prefix = str(prefix).rstrip(".")
    selected = [
        name for name in state_dict
        if name == prefix or name.startswith(prefix + ".")
    ]
    if not selected:
        raise ValueError(f"no parameters under personalization prefix {prefix!r}")
    return selected


def _last_weight_prefix(state_dict):
    weights = [name for name in state_dict if name.endswith("weight")]
    if not weights:
        raise ValueError("state has no weight tensors to personalize")
    return weights[-1][: -len("weight")].rstrip(".")


class FedPerClientTrainer(ClientTrainer):
    """Train the full model. Upload the base with the global head put back.

    ``personal_state`` is the model the client deploys: the round's base plus
    its own head. ``evaluate_personalized_local`` already reads that attribute.
    """

    def __init__(self, *args, personal_prefix=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.personal_prefix = personal_prefix
        self.personal_state = None
        self._upload_state = None

    def download(self, state_dict):
        super().download(state_dict)
        if self.personal_state is None:
            return self
        overlay = {
            key: self.personal_state[key]
            for key in personalization_keys(state_dict, self.personal_prefix)
        }
        self.model.load_state_dict(overlay, strict=False)
        return self

    def train(self, global_state=None):
        if global_state is None:
            global_state = deepcopy(self.model.state_dict())
        super().train(global_state=global_state)
        trained = deepcopy(self.model.state_dict())
        self.personal_state = trained
        uploaded = deepcopy(trained)
        for key in personalization_keys(trained, self.personal_prefix):
            uploaded[key] = global_state[key].detach().clone()
        self._upload_state = uploaded
        self.model.load_state_dict(uploaded)
        return self

    def upload(self):
        if self._upload_state is None:
            return super().upload()
        return deepcopy(self._upload_state)
