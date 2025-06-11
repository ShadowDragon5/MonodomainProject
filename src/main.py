import torch


def main():
    print(torch.cuda.is_available())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping...")
