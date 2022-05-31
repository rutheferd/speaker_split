import logging

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def there(name, greeting=False):
    sentence = ""
    if greeting:
        logging.info("Generating a sentence with a greeting...")
        sentence = "Hello there {}, how are you?".format(name)
    else:
        logging.info("Generating a sentence with just a name...")
        sentence = "Hello there {}.".format(name)
    return sentence


def run(name, greeting):
    sentence = there(name, greeting)
    print(sentence)
