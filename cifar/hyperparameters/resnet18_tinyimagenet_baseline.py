config = dict(
    dataset="tinyimagenet",
    model="resnet18",
    optimizer="SGD",
    optimizer_decay_at_epochs=[30, 60, 90, 120],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.1,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0001,
    batch_size=256,
    num_epochs=120,
    seed=42,
)
