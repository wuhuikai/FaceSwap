from PIL import Image, ImageDraw, ImageFont


class MemeGenerator:
    def __init__(self, font="Impact.ttf"):
        self.font_name = font

    def generate_text_array(self, font, text, img_width, font_cushion):
        font_size = font.getsize(text)
        texts = []

        # If the text is larger than the width
        if font_size[0] > img_width:
            all_words = text.split(sep=" ")
            words = ""
            for cnt, word in enumerate(all_words):

                # If we're in the first entry
                if len(words) > 0:
                    words = words + " "

                # If adding the new word puts us bigger than the width
                if font.getsize(words + word)[0] > img_width - font_cushion:
                    words = words.strip()
                    texts.append(words)
                    words = ""  # Clear out the words for the new line

                # Add a new word to the list
                words = words + word

            # Add left over words
            words.strip()
            texts.append(words)
        else:
            texts = [text]
        return texts

    def _write_meme(self, draw, top_x, border_size, y, text, font, fill_color="white", shadow_color="black"):

        # Thin text border
        draw.text((top_x - border_size, y), text, font=font, fill=shadow_color)
        draw.text((top_x + border_size, y), text, font=font, fill=shadow_color)
        draw.text((top_x, y - border_size), text, font=font, fill=shadow_color)
        draw.text((top_x, y + border_size), text, font=font, fill=shadow_color)

        # Draw text border
        draw.text((top_x - border_size, y - border_size), text, font=font, fill=shadow_color)
        draw.text((top_x + border_size, y - border_size), text, font=font, fill=shadow_color)
        draw.text((top_x - border_size, y + border_size), text, font=font, fill=shadow_color)
        draw.text((top_x + border_size, y + border_size), text, font=font, fill=shadow_color)

        # now draw the text over it
        draw.text((top_x, y), text, font=font, fill=fill_color)

    def generate_meme(self, image_source, text_top="", text_bottom=""):
        # Load the image
        image = Image.open(image_source)
        draw = ImageDraw.Draw(image)

        # Set font dimensions
        font_height = int(min(image.height, image.width) / 8.5)
        font_cushion = font_height / 20
        border_size = int(font_height / 20)

        # Use a TrueType font
        font = ImageFont.truetype(self.font_name, font_height)
        font_height = font.getsize(text_top + text_bottom)[1]

        if len(text_top) > 0:
            texts_top = self.generate_text_array(font, text_top, image.width, font_cushion)
            y_start = font_cushion
            for cnt, text in enumerate(texts_top):
                font_width = font.getsize(text)[0]
                top_x = (image.width - font_width) / 2  # Calculate the starting location of the text (x)
                y = y_start + cnt * font_height  # Calculate the starting location
                self._write_meme(draw, top_x, border_size, y, text, font)  # Draw the text out

        if len(text_top) > 0:
            texts_bottom = self.generate_text_array(font, text_bottom, image.width, font_cushion)
            texts_bottom.reverse()
            y_start = image.height - font_cushion - font_height
            for cnt, text in enumerate(texts_bottom):
                font_width = font.getsize(text)[0]
                top_x = (image.width - font_width) / 2  # Calculate the starting location of the text (x)
                y = y_start - cnt * font_height - (border_size * 5)  # Calculate the starting location
                self._write_meme(draw, top_x, border_size, y, text, font)  # Draw the text out

        return image
