import os
import re
import time
import datetime
import random
import json
from io import BytesIO
from mimetypes import MimeTypes
from pathlib import Path
import folder_paths
import magic
from PIL import Image

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')

class TextTokens:
    def __init__(self):
        self.tokens = {
            '[time]': str(time.time()).replace('.', '_')
        }
        if '.' in self.tokens['[time]']:
            self.tokens['[time]'] = self.tokens['[time]'].split('.')[0]

    def format_time(self, format_code):
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text):
        tokens = self.tokens.copy()
        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            text = text.replace(token, value)

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, text)
        return text

def parse_output_path_base(output_path, base_dir):
    tokens = TextTokens()
    if output_path in [None, '', "none", "."]:
        return base_dir
    output_path = tokens.parseTokens(output_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
    return output_path


def append_subdirs_before_stem(output_path, subdirs):
    path = Path(output_path)
    valid = [str(s) for s in subdirs if s and str(s) != 'None' and str(s).strip()]
    if valid:
        return os.path.join(str(path.parent), *valid, path.stem)
    return os.path.join(str(path.parent), path.stem)

def ensure_output_dir(output_path):
    if output_path.strip():
        if not os.path.isabs(output_path):
            output_path = os.path.join(folder_paths.output_directory, output_path)
        if not os.path.exists(output_path.strip()):
            print(f"The path `{output_path.strip()}` specified doesn't exist! Creating directory.")
            os.makedirs(output_path, exist_ok=True)
    return output_path

def build_filename_and_counter(output_path, prefix, delimiter, number_padding, number_start, extension, overwrite_mode='false'):
    number_start_bool = number_start is True or str(number_start).lower() == 'true'
    file_extension = ('.' + extension) if not extension.startswith('.') else extension
    if file_extension not in ALLOWED_EXT:
        file_extension = '.jpg'
    if number_start_bool:
        pattern = f"(\\d{{{number_padding}}}){re.escape(delimiter)}{re.escape(prefix)}"
    else:
        pattern = f"{re.escape(prefix)}{re.escape(delimiter)}(\\d{{{number_padding}}})"
    existing_counters = [
        int(re.search(pattern, fn).group(1))
        for fn in os.listdir(output_path)
        if re.match(pattern, os.path.basename(fn))
    ]
    existing_counters.sort(reverse=True)
    counter = existing_counters[0] + 1 if existing_counters else 1
    if overwrite_mode == 'prefix_as_filename':
        file = f"{prefix}{file_extension}"
    else:
        if number_start_bool:
            file = f"{counter:0{number_padding}}{delimiter}{prefix}{file_extension}"
        else:
            file = f"{prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
        if os.path.exists(os.path.join(output_path, file)):
            counter += 1
            if number_start_bool:
                file = f"{counter:0{number_padding}}{delimiter}{prefix}{file_extension}"
            else:
                file = f"{prefix}{delimiter}{counter:0{number_padding}}{file_extension}"

    return file, counter

def resolve_output_file(output_path_input, base_dir, subdirs, prefix, delimiter, add_date, add_time, number_padding, number_start, extension):
    resolved = parse_output_path_base(output_path_input, base_dir)
    resolved = append_subdirs_before_stem(resolved, subdirs)
    resolved = ensure_output_dir(resolved)
    tokens = TextTokens()
    prefix_parsed = tokens.parseTokens(prefix)
    nowdate = datetime.datetime.now()
    if add_date:
        prefix_parsed = prefix_parsed + delimiter + nowdate.strftime("%Y-%m-%d")
    if add_time:
        prefix_parsed = prefix_parsed + delimiter + nowdate.strftime("%H%M%S")

    filename, counter = build_filename_and_counter(
        output_path=resolved,
        prefix=prefix_parsed,
        delimiter=delimiter,
        number_padding=number_padding,
        number_start=number_start,
        extension=extension,
    )

    output_file = os.path.abspath(os.path.join(resolved, filename))
    json_file = os.path.splitext(output_file)[0] + '.json'
    txt_file = os.path.splitext(output_file)[0] + '.txt'

    return output_file, json_file, txt_file


def detect_mime(save_bytes, temp_directory):
    tmp_path = os.path.join(temp_directory, f"api_mime_{random.randint(1000, 9999)}")
    with open(tmp_path, 'wb') as f:
        f.write(save_bytes)
    try:
        mime = magic.from_file(tmp_path, mime=True)
    except Exception:
        mime, _ = MimeTypes().guess_type(tmp_path)
    return mime or 'application/octet-stream'

def save_bytes_to_file(save_bytes, output_file, image_extension, image_quality, temp_directory):
    if save_bytes is None:
        return output_file
    mime = detect_mime(save_bytes, temp_directory)
    stem = os.path.splitext(output_file)[0]
    if mime.startswith('image/'):
        fmt_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'webp': 'WEBP', 'png': 'PNG', 'tiff': 'TIFF', 'gif': 'GIF'}
        fmt = fmt_map.get(image_extension.lower(), image_extension.upper())
        img = Image.open(BytesIO(save_bytes))
        if fmt in ('PNG', 'TIFF', 'GIF'):
            img.save(output_file, format=fmt)
        elif fmt == 'JPEG':
            img.save(output_file, format='JPEG', quality=image_quality, optimize=True)
        else:
            img.save(output_file, format=fmt, quality=image_quality)
    elif mime.startswith('audio/'):
        output_file = stem + '.mp3'
        with open(output_file, 'wb') as f:
            f.write(save_bytes)
    elif mime.startswith('video/'):
        output_file = stem + '.mp4'
        with open(output_file, 'wb') as f:
            f.write(save_bytes)
    elif mime.startswith('text/'):
        output_file = stem + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(save_bytes.decode('utf-8', errors='replace'))
    else:
        with open(output_file, 'wb') as f:
            f.write(save_bytes)
    return output_file

def save_metadata(save_data, json_file, txt_file, save_data_to_json, save_data_to_txt, used_values):
    if save_data_to_json:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
    if save_data_to_txt:
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"provider: {save_data.get('provider', '')}\n")
            f.write(f"service: {save_data.get('service', '')}\n")
            if isinstance(used_values, dict):
                for k, v in flatten_dict(used_values):
                    f.write(f"{k}: {v}\n")

def sanitize_path_part(value):
    result = re.sub(r'[ /\\.,;`\-]+', '_', str(value))
    return re.sub(r'_+', '_', result).strip('_')

def flatten_dict(d, prefix=''):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from flatten_dict(v, key)
        else:
            yield key, v
