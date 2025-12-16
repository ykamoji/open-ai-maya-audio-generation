import os
import glob
import shutil
import re
import json
import argparse
from PostProcess.audio_extraction import extract_all_features, extract_audio_features
from PostProcess.image_extraction import extract_image_data


def move_audio_and_subtitle_files(source_base_path: str, destination_folder: str):
    """
    Finds all .m4a and .srt files within specified subdirectories of the source path
    and moves them to the destination folder.

    Args:
        source_base_path (str): The base path where the 'audios' subdirectory is located.
                                  e.g., '/content/drive/MyDrive/AI/Outputs/'
        destination_folder (str): The target folder to move the files to.
                                  e.g., '/content/drive/MyDrive/AI/Audiobook'
    """
    os.makedirs(destination_folder, exist_ok=True)
    print(f"Ensured existence of destination folder: {destination_folder}")

    files_to_move = []
    for ext in ['m4a', 'srt']:
        # Correct glob pattern to look for .wav and .srt files in immediate subdirectories of 'audios'
        search_pattern = os.path.join(source_base_path, f"audios/*/*.{ext}")
        files_to_move.extend(glob.glob(search_pattern))

    if not files_to_move:
        print(f"No .wav or .srt files found in {os.path.join(source_base_path, 'audios/*/')}")
        return

    print(f"Found {len(files_to_move)} files to move.")
    moved_count = 0
    for f in files_to_move:
        try:
            shutil.move(f, os.path.join(destination_folder, os.path.basename(f)))
            print(f"Moved {os.path.basename(f)} to {destination_folder}")
            moved_count += 1
        except Exception as e:
            print(f"Error moving file {f}: {e}")

    print(f"Successfully moved {moved_count} out of {len(files_to_move)} files to {destination_folder}")


def read_chapter_intro(srt_path):
    with open(srt_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    subtitle_lines = []
    collecting = False

    for line in lines:
        line = line.strip()

        # Skip index and timestamp lines
        if not line:
            if collecting:
                break
            continue
        if "-->" in line or line.isdigit():
            continue

        collecting = True
        subtitle_lines.append(line)

    return " ".join(subtitle_lines)


def get_audio_titles(path):

    intro_map = {}

    srt_files = glob.glob(os.path.join(path, f"*.srt"))
    srt_files.sort(key=lambda x: getChapterNo(x))

    for srt_file in srt_files:
        intro_map[os.path.splitext(os.path.basename(srt_file))[0]] = read_chapter_intro(srt_file)

    return intro_map


def getChapterNo(path):
    return int(re.search(r'\d+', path).group())


def create_audio_extraction_data(path):

    audio_files = glob.glob(os.path.join(path, f"*.m4a"))
    audio_files.sort(key=lambda x: getChapterNo(x))

    srt_files = glob.glob(os.path.join(path, f"*.srt"))
    srt_files.sort(key=lambda x: getChapterNo(x))

    return audio_files, srt_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ColorScheme")

    parser.add_argument("--s", type=str,
                        default="output", help="Audio source Paths")
    parser.add_argument("--d", type=str,
                        default="/Users/ykamoji/Documents/Audiobooks", help="Audio destination Paths")
    args = parser.parse_args()

    move_audio_and_subtitle_files(args.s, args.d)

    audio_paths, srt_paths = create_audio_extraction_data(args.d)

    metadata = {
        "audiobook_progress":{},
        "static":{},
        "audiobook_playlists":[]
    }

    if os.path.isfile(f"{args.d}/metadata.json"):
        with open(f"{args.d}/metadata.json") as f:
            metadata = json.load(f)

    processed_aud_paths = audio_paths[:]
    processed_srt_paths = srt_paths[:]
    for chapter in metadata["static"]:
        for audio_path in audio_paths:
            if chapter in os.path.splitext(os.path.basename(audio_path))[0] and "prosody_index" in metadata["static"][chapter]:
                processed_aud_paths.remove(audio_path)

        for srt_path in srt_paths:
            if chapter in os.path.splitext(os.path.basename(srt_path))[0] and "speed" in metadata["static"][chapter]:
                processed_srt_paths.remove(srt_path)

    # for (a, s) in prepared_data:
    #     print(a, s)
    #     audio_data = extract_audio_features(a, s)

    if processed_aud_paths:
        prepared_data = zip(processed_aud_paths, processed_srt_paths)

        audio_data = extract_all_features(prepared_data)

        for audio_name in audio_data:
            metadata["static"][audio_name] = audio_data[audio_name]

        with open(f"{args.d}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    image_data = extract_image_data(args.d)

    for image_name in image_data:
        data = metadata["static"].get(image_name, {})
        data['scheme'] = image_data[image_name]['scheme']
        data['dims'] = image_data[image_name]['dims']
        metadata["static"][image_name] = data

    intro_data = get_audio_titles(args.d)
    for title in intro_data:
        data = metadata["static"].get(title, {})
        data['intro'] = intro_data[title]
        metadata["static"][title] = data

    with open(f"{args.d}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)




