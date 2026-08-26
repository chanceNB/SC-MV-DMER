"""Training components with explicit research-contract boundaries."""

from sc_mv_dmer.training.sensor_executor import (
    SensorFormalExecutor,
    SensorLossContract,
    SensorOnlyModel,
    TrainingExecutionNotAuthorized,
)

__all__ = ["SensorOnlyModel", "SensorLossContract", "SensorFormalExecutor", "TrainingExecutionNotAuthorized"]
