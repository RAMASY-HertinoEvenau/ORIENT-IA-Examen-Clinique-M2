"""Point d'entrée local de vérification du noyau ORIENT'IA."""

from orient_ia import __version__


def main() -> None:
    print(f"ORIENT'IA {__version__} - noyau installé")


if __name__ == "__main__":
    main()