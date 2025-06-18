import numpy as np

def simple_portfolio_value_change(current_portfolio_value: float, previous_portfolio_value: float) -> float:
    """
    Calculates the reward based on the change in portfolio value.

    Args:
        current_portfolio_value: The portfolio value at the current step.
        previous_portfolio_value: The portfolio value at the previous step.

    Returns:
        The calculated reward.
    """
    return current_portfolio_value - previous_portfolio_value

def loss_based_reward(current_loss: float, previous_loss: float) -> float:
    """
    Calculates reward based on the change in a loss function.
    A decrease in loss is positive reward.

    Args:
        current_loss: The loss at the current step.
        previous_loss: The loss at the previous step.

    Returns:
        The calculated reward.
    """
    return previous_loss - current_loss
