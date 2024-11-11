# Machine Learning Concepts

## Epoch
- One full run through the entire dataset during training.
- **First epoch**: Initializes and adjusts connections (weights).
- **Subsequent epochs**: Refine connections by reinforcing or weakening them based on learning.

## label_mode
- Specifies the format of labels in the dataset.
    - **`"int"`**: Integer labels (e.g., 0, 1, 2).
    - **`"categorical"`**: One-hot encoded labels (e.g., [0, 1, 0]).
    - **`"binary"`**: Single binary values (e.g., 0 or 1).
- Use `"categorical"` for multi-class image recognition tasks.

## Activation
- Functions that introduce non-linearity to the model, enabling it to learn complex patterns.
    - **ReLU**: \( \max(0, x) \), common in hidden layers.
    - **Softmax**: Converts outputs into probabilities for multi-class classification.
    - **Sigmoid**: Squashes values to (0, 1), often for binary classification.
    - **Tanh**: Outputs (-1, 1), less common than ReLU.

## Validation
- Evaluates the model's performance on unseen data during training.
- Checks for overfitting or underfitting.
- Does **not** update weights; purely for monitoring generalization.

## Optimizers
- Algorithms that adjust weights to minimize loss.
    - **SGD**: Simple, effective gradient descent.
    - **Adam**: Adaptive optimizer, fast and widely used.
    - **RMSProp**: Good for RNNs and noisy data.
- Adam is the default choice for most tasks.

## Regularization
- Techniques to prevent overfitting by discouraging model complexity.
    - **L1**: Adds absolute value of weights to the loss, encourages sparsity.
    - **L2**: Adds squared weights to the loss, smooths the model.
    - **Dropout**: Randomly disables neurons during training.
    - **Early Stopping**: Stops training when validation performance stagnates.
- Combine methods (e.g., L2 + Dropout) for better generalization.

## Learning Rate
- Controls the step size in weight updates during optimization.
- **Too high**: May cause overshooting and fail to converge.
- **Too low**: May make convergence very slow.
- Use a **learning rate scheduler** to adapt the learning rate over epochs.

## Batch Size
- The number of samples processed before updating weights.
    - **Small batch sizes**: More updates, slower training but can generalize better.
    - **Large batch sizes**: Faster training, but may require more memory and risk overfitting.

## Loss Function
- Measures the difference between predictions and actual target values.
    - **Mean Squared Error (MSE)**: Regression tasks.
    - **Binary Crossentropy**: Binary classification.
    - **Categorical Crossentropy**: Multi-class classification.
    - **Huber Loss**: Robust to outliers in regression tasks.
- The choice of loss depends on the problem type.

## Metrics
- Used to evaluate the performance of the model during training and validation.
    - **Accuracy**: Percentage of correct predictions (common for classification).
    - **Precision/Recall/F1-Score**: Important for imbalanced datasets.
    - **Mean Absolute Error (MAE)**: Average absolute error (used in regression).

## Data Preprocessing
- Ensures data is clean and ready for training.
    - **Normalization**: Scale inputs to a standard range (e.g., [0, 1]).
    - **Standardization**: Scale inputs to have zero mean and unit variance.
    - **Data Augmentation**: Generate new data by transforming existing samples (e.g., flipping, rotating).

## Callback Functions
- Functions that modify the training process at runtime.
    - **Early Stopping**: Stops training when validation loss stops improving.
    - **Model Checkpoint**: Saves the model at certain intervals.
    - **Learning Rate Scheduler**: Adjusts the learning rate dynamically.
    - **TensorBoard**: Visualizes metrics and graphs during training.

## Overfitting vs. Underfitting
- **Overfitting**: Model performs well on training data but poorly on unseen data.
    - Solution: Use regularization, dropout, or data augmentation.
- **Underfitting**: Model fails to capture patterns in the training data.
    - Solution: Use a more complex model or train for more epochs.

## Test Set
- A separate dataset used to evaluate the final performance of the model after training.
- Must not be used during training or validation to avoid bias.
