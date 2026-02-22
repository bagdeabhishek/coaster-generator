import math
import io
from PIL import Image, ImageDraw, ImageFont
import os

def draw_curved_text(img, text, center, radius, font):
    """Draws curved text along an arc using precise character widths."""
    total_chars = len(text)
    if total_chars == 0:
        return
        
    # Get exact width of each character to calculate proportional angles
    char_widths = []
    dummy_img = Image.new('RGBA', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    total_width = 0
    for char in text:
        bbox = dummy_draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        char_widths.append(w)
        total_width += w
        
    # Add spacing between characters
    spacing_ratio = 0.2 # 20% of average char width as space
    avg_width = total_width / total_chars if total_chars > 0 else 0
    space_width = avg_width * spacing_ratio
    
    # Calculate circumference
    circumference = 2 * math.pi * radius
    
    # Calculate how much angle the whole text takes
    total_text_width = total_width + (space_width * (total_chars - 1))
    total_angle_deg = (total_text_width / circumference) * 360
    
    # Start angle (centered at top, which is -90 degrees)
    current_angle_deg = -90 - (total_angle_deg / 2)
    
    for i, char in enumerate(text):
        # Calculate the angle for the center of this character
        char_angle_deg = current_angle_deg + ((char_widths[i] / circumference) * 360 / 2)
        angle_rad = math.radians(char_angle_deg)
        
        # Calculate position on the circle
        x = center[0] + radius * math.cos(angle_rad)
        y = center[1] + radius * math.sin(angle_rad)
        
        # Create a tiny transparent image for the character
        char_img = Image.new('RGBA', (font.size * 3, font.size * 3), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        
        # Draw text exactly in the center
        char_draw.text((font.size * 1.5, font.size * 1.5), char, font=font, fill="black", anchor="mm")
        
        # Rotate the character so its bottom points to the center
        rotated_char = char_img.rotate(-char_angle_deg - 90, expand=True, resample=Image.BICUBIC)
        
        # Paste the rotated character onto the main image
        rx, ry = rotated_char.size
        paste_x = int(x - rx/2)
        paste_y = int(y - ry/2)
        
        # Simple thresholding to clean up rotation artifacts
        data = rotated_char.getdata()
        new_data = []
        for item in data:
            if item[3] > 128:
                new_data.append((0, 0, 0, 255))
            else:
                new_data.append((0, 0, 0, 0))
        rotated_char.putdata(new_data)
        
        img.paste(rotated_char, (paste_x, paste_y), rotated_char)
        
        # Advance the angle for the next character
        current_angle_deg += ((char_widths[i] + space_width) / circumference) * 360

def composite_coaster_image(portrait_bytes: bytes, stamp_text: str) -> bytes:
    import io
    
    # 1. Load the AI generated portrait (which should be 512x512)
    portrait_img = Image.open(io.BytesIO(portrait_bytes)).convert("RGBA")
    
    # 2. Create the high-res 1024x1024 canvas for crisp vectorization
    canvas_size = 1024
    center = (canvas_size // 2, canvas_size // 2)
    img = Image.new("RGBA", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(img)
    
    # Dimensions for rings
    outer_radius = 480
    inner_radius = 360
    ring_width = 16
    
    # 3. Draw outer and inner rings
    draw.ellipse(
        [(center[0]-outer_radius, center[1]-outer_radius), 
         (center[0]+outer_radius, center[1]+outer_radius)], 
        outline="black", width=ring_width
    )
    
    draw.ellipse(
        [(center[0]-inner_radius, center[1]-inner_radius), 
         (center[0]+inner_radius, center[1]+inner_radius)], 
        outline="black", width=ring_width
    )
    
    # 4. Paste and scale portrait into the center
    portrait_target_size = int(inner_radius * 2 * 0.95) 
    portrait_resized = portrait_img.resize((portrait_target_size, portrait_target_size), Image.Resampling.LANCZOS)
    
    # Make white pixels transparent
    data = portrait_resized.getdata()
    new_data = []
    for item in data:
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    portrait_resized.putdata(new_data)
    
    paste_x = center[0] - portrait_target_size // 2
    paste_y = center[1] - portrait_target_size // 2
    img.paste(portrait_resized, (paste_x, paste_y), portrait_resized)
    
    # 5. Draw the curved text
    if stamp_text:
        try:
            import os
            font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts", "Roboto-Bold.ttf")
            font = ImageFont.truetype(font_path, 80)
        except IOError:
            print("Warning: Roboto-Bold not found, using default")
            font = ImageFont.load_default()
            
        text_radius = (outer_radius + inner_radius) / 2
        draw_curved_text(img, stamp_text.upper(), center, text_radius, font)
    
    # 6. Convert to RGB and return bytes
    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
    rgb_img.paste(img, mask=img.split()[3])
    
    output_io = io.BytesIO()
    rgb_img.save(output_io, format="PNG")
    return output_io.getvalue()
