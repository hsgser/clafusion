config = dict(
    dataset="esc50",
    model="resnet34_nobias_nobn",
    optimizer="SGD",
    optimizer_decay_at_epochs=[30, 60, 90, 120],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.01,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0001,
    batch_size=8,
    num_epochs=120,
    test_fold=5,
    seed=42,
)
