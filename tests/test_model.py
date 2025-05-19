import torch
import pytest
from neurogrow.model import DynamicWidthCNNLightning


@pytest.fixture
def model():
    """Fixture that provides a model instance for testing."""
    return DynamicWidthCNNLightning(
        initial_width=16, learning_rate=0.001, width_double_epoch=2
    )


def test_model_initialization(model):
    """Test model initialization with different widths."""
    assert model.width == 16
    assert model.conv1.out_channels == 16
    assert model.conv2.out_channels == 32
    assert model.conv3.out_channels == 64
    assert model.fc2.out_features == 1000  # ImageNet classes


def test_model_forward_pass(model):
    """Test that the forward pass works and returns correct shape."""
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)  # ImageNet image size
    output = model(input_tensor)

    # Check output shape (batch_size, num_classes)
    assert output.shape == (batch_size, 1000)
    # Check that output contains finite values
    assert torch.isfinite(output).all()


def test_model_width_doubling(model):
    """Test the width doubling functionality."""
    initial_width = model.width

    # Store initial weights for comparison
    initial_conv1_weight = model.conv1.weight.clone()
    initial_conv1_bias = model.conv1.bias.clone()

    # Double the width
    model.double_width()

    # Check that width has doubled
    assert model.width == initial_width * 2
    assert model.conv1.out_channels == initial_width * 2
    assert model.conv2.out_channels == initial_width * 4
    assert model.conv3.out_channels == initial_width * 8

    # Check that the original weights are preserved in the new layers (first initial_width channels)
    assert torch.allclose(model.conv1.weight[:initial_width], initial_conv1_weight)
    assert torch.allclose(model.conv1.bias[:initial_width], initial_conv1_bias)

    # Check that new weights are initialized (not zero)
    assert not torch.allclose(
        model.conv1.weight[initial_width:],
        torch.zeros_like(model.conv1.weight[initial_width:]),
    )
    assert not torch.allclose(
        model.conv1.bias[initial_width:],
        torch.zeros_like(model.conv1.bias[initial_width:]),
    )


def test_model_forward_after_width_doubling(model):
    """Test that forward pass still works after width doubling."""
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    # Get output before width doubling
    output_before = model(input_tensor)

    # Double width
    model.double_width()

    # Get output after width doubling
    output_after = model(input_tensor)

    # Check that outputs have same shape
    assert output_before.shape == output_after.shape
    # Check that outputs are different (due to new random weights)
    assert not torch.allclose(output_before, output_after)


def test_model_device_transfer(model):
    """Test that model can be moved to different devices."""
    if torch.cuda.is_available():
        # Test moving to GPU
        model_gpu = model.cuda()
        assert next(model_gpu.parameters()).is_cuda

        # Test moving back to CPU
        model_cpu = model_gpu.cpu()
        assert not next(model_cpu.parameters()).is_cuda
    else:
        # If no GPU available, just test CPU
        assert not next(model.parameters()).is_cuda


def test_output_consistency_after_growing():
    """Test that output is exactly the same after growing and copying weights, with new weights/biases zeroed."""
    torch.manual_seed(0)
    model = DynamicWidthCNNLightning(initial_width=8)
    model.eval()
    input_tensor = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output_before = model(input_tensor)
    # Save original weights and biases
    orig_conv1_weight = model.conv1.weight.data.clone()
    orig_conv1_bias = model.conv1.bias.data.clone()
    orig_conv2_weight = model.conv2.weight.data.clone()
    orig_conv2_bias = model.conv2.bias.data.clone()
    orig_conv3_weight = model.conv3.weight.data.clone()
    orig_conv3_bias = model.conv3.bias.data.clone()
    orig_fc1_weight = model.fc1.weight.data.clone()
    orig_fc1_bias = model.fc1.bias.data.clone()
    # Grow the model
    model.double_width()
    # Overwrite the new weights with the originals in the corresponding positions
    with torch.no_grad():
        # conv1
        model.conv1.weight.data[:8] = orig_conv1_weight
        model.conv1.bias.data[:8] = orig_conv1_bias
        model.conv1.weight.data[8:] = 0
        model.conv1.bias.data[8:] = 0
        # conv2
        model.conv2.weight.data[:16, :8] = orig_conv2_weight
        model.conv2.bias.data[:16] = orig_conv2_bias
        model.conv2.weight.data[:16, 8:] = 0
        model.conv2.weight.data[16:] = 0
        model.conv2.bias.data[16:] = 0
        # conv3
        model.conv3.weight.data[:32, :16] = orig_conv3_weight
        model.conv3.bias.data[:32] = orig_conv3_bias
        model.conv3.weight.data[:32, 16:] = 0
        model.conv3.weight.data[32:] = 0
        model.conv3.bias.data[32:] = 0
        # fc1
        model.fc1.weight.data[:, : 32 * 28 * 28] = orig_fc1_weight
        model.fc1.weight.data[:, 32 * 28 * 28 :] = 0
        model.fc1.bias.data = orig_fc1_bias
    model.eval()
    with torch.no_grad():
        output_after = model(input_tensor)
    # The outputs should be exactly the same
    assert torch.allclose(output_before, output_after, atol=1e-6)
