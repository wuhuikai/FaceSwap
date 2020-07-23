#! /usr/bin/env python
import argparse
import sys

from meme.meme_generator import MemeGenerator


def main():
    parser = argparse.ArgumentParser(description="Meme")
    parser.add_argument("--src", help="Path for source image", default="../imgs/test6.jpg")
    parser.add_argument("--out", help="Path for storing output images", default="../imgs/face_out.png")
    args = parser.parse_args()

    meme = MemeGenerator()
    image = meme.generate_meme(image_source=args.src, text_top="Hello", text_bottom="World",)
    image.save(args.out)
    return


if __name__ == "__main__":
    sys.exit(main())
