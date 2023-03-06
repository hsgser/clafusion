config = dict(
    dataset="Cifar10",
    model="vgg13_student_nobias",
    optimizer="SGD",
    optimizer_decay_at_epochs=[20, 40, 60, 80, 100],
    optimizer_decay_with_factor=2.0,
    optimizer_learning_rate=0.01,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0005,
    batch_size=128,
    num_epochs=120,
    seed=42,
)
