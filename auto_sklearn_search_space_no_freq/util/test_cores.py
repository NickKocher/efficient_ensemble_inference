import os


def main():
    print(os.sched_getaffinity(0))
    return 0


if __name__ == "__main__":
    main()
