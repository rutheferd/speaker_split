from click.testing import CliRunner
from elmer.src import train
from elmer.__main__ import main


def test_there_function():
    name = "Austin"
    greeting = True
    assert (
        train.there(name, greeting)
        == "Hello there Austin, how are you?"
    )

    name = "Austin"
    assert train.there(name) == "Hello there Austin."

    name = "Austin"
    greeting = False
    assert train.there(name, greeting) == "Hello there Austin."

    pass


def test_there_command():
    # Testing with greeting
    name = "Austin"
    runner = CliRunner()
    result = runner.invoke(main, ["there", "-n", name, "-g"])
    assert result.output == "Hello there Austin, how are you?\n"

    # Testing without greeting
    name = "Obi-Wan"
    result = runner.invoke(main, ["there", "-n", name])
    assert result.output == "Hello there Obi-Wan.\n"

    pass
