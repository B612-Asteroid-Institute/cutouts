import argparse

def main():

    my_parser = argparse.ArgumentParser(
        prog="cutouts",
        description="Get and plot cutouts along a trajectory."
    )
    my_parser.parse_args()