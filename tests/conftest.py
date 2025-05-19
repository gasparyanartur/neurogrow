import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
