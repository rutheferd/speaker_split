import click
from elmer.src import train


@click.group()
@click.version_option(package_name="hello_template")
def main():
    """Hello is a CLI tool for creating a custom greeting to send to friend."""
    pass


# Note below that I am only using options, the click.argument can also be used but has limited capability.
# I generally like to use click.option and setting the option flag to required, as seen in the name option.
@click.option(
    # Right now we will assume that the user will follow the README
    # and create the appropriate folder structure.
    "--data_path",
    "-d",
    type=click.STRING,
    required=True,
    help="Path to the dataset to train on.",
)
@click.option(
    # May want to group.
    "--val_split",
    "-vs",
    type=click.FLOAT,
    default=0.1,
    help="Percentage of samples to use fo validation.",
)
@click.option(
    # May want to group.
    "--sampling_rate",
    "-sr",
    type=click.INT,
    default=16000,
    help="The sampling rate to use.",
)
@click.option(
    # May want to group.
    "--batch_size",
    "-bs",
    type=click.INT,
    default=128,
    help="Batch Size for Training.",
)
@click.option(
    # May want to group.
    "--num_epochs",
    "-ne",
    type=click.INT,
    default=100,
    help="Number of Training Epochs.",
)
@main.command()
def train(data_path, val_split, sampling_rate, batch_size, num_epochs):
    """Create a Greeting to Send to a Friend!"""
    train.run(data_path, val_split, sampling_rate, batch_size, num_epochs)


if __name__ == "__main__":
    main()
