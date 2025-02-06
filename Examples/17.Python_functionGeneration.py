# Making double sure, that function generation works as I imagine.


def generate_function(i: int):
    def function() -> None:
        return i

    return function


def main():
    functions = []
    for i in range(9):
        functions.append(generate_function(i))

    for i in range(9):
        assert functions[i]() == i

    print("\tTest done.")


if __name__ == "__main__":
    print("Experiment start!")
    main()
    print("Experiment finished.")
