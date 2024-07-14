from monai.config import IgniteInfo
from monai.utils import min_version, optional_import
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import torch

reinit__is_reduced, _ = optional_import(
    "ignite.metrics.metric", IgniteInfo.OPT_IMPORT_VERSION, min_version, "reinit__is_reduced"
)
if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")


class MeanLoss(Metric):
    r"""
    Computes Dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """
    def __init__(
        self, output_transform: Callable = lambda x: x) -> None:
        """[summary]

        Args:
            output_transform (Callable, optional): [description]. Defaults to lambdax:x.
        """        ''''''
        self.metric_fn = lambda x: x
        self.losses = []
        super().__init__(output_transform=output_transform,)

    @reinit__is_reduced
    def reset(self) -> None:
        self.losses = []

    @reinit__is_reduced
    def update(self, output) -> None:
        """[summary]

        Args:
            output ([type]): [description]

        Returns:
            [type]: [description]
        """        ''''''
        self.losses.append(output)

    def compute(self) -> Any:
        """[summary]

        Raises:
            RuntimeError: [description]

        Returns:
            Any: [description]
        """        ''''''
        return torch.mean(torch.tensor(self.losses))

    def attach(self, engine: Engine, name: str) -> None:
        """[summary]

        Args:
            engine (Engine): [description]
            name (str): [description]
        """        ''''''
        super().attach(engine=engine, name=name)
        # FIXME: record engine for communication, ignite will support it in the future version soon
        self._engine = engine
        self._name = name
        if self.save_details and not hasattr(engine.state, "metric_details"):
            engine.state.metric_details = {}
