from tinygpt.tensor import Tensor
from tinygpt.module import Module


class CrossEntropyLoss(Module):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, logits: Tensor, labels: Tensor) -> Tensor:
        if not 0 < logits.ndim < 3:
            raise RuntimeError(f"logits dimension should be 1 or 2 but found shape {logits.shape}")
        if not 0 < labels.ndim < 3:
            raise RuntimeError(f"labels dimension should be 1 or 2 but found shape {labels.shape}")

        log_probs = logits.softmax(axis=len(logits.shape) - 1).log()
        loss = -((log_probs * labels).sum(axes=(len(log_probs.shape) - 1,), keepdim=False))

        return loss